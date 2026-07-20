"""
Computes the structured_facts feature set for each alert.
These are the same kind of features that later feed the Phase 2 ranking model.
"""
import numpy as np
import pandas as pd


def _prior_expanding_mean_std(transactions: pd.DataFrame, group_col: str, val_col: str):
    """
    Vectorized prior-only (shift-1) expanding mean/std per group via cumulative
    sums, replacing groupby.transform(lambda s: s.shift(1).expanding()...). The
    lambda form runs Python per group and dominates runtime at millions of
    accounts; this yields identical statistics with vectorized cumsum passes.
    Matches pandas semantics: mean NaN with 0 prior points, std (ddof=1) NaN with
    fewer than 2. Assumes transactions is sorted by (group_col, ts).
    """
    vals = transactions[val_col].astype(float)
    keys = transactions[group_col]
    prior_count = vals.groupby(keys, sort=False).cumcount().astype(float)
    prior_sum = vals.groupby(keys, sort=False).cumsum() - vals
    prior_sumsq = (vals * vals).groupby(keys, sort=False).cumsum() - vals * vals

    n = prior_count.replace(0, np.nan)
    mean = prior_sum / n
    denom = (prior_count - 1).where(prior_count >= 2, np.nan)
    var = (prior_sumsq - (prior_sum * prior_sum) / n) / denom
    std = np.sqrt(var.clip(lower=0))
    return mean, std


def compute_velocity_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    24h trailing transaction count and sum, per account, at the time of each
    transaction.

    NOTE: kept in (account_id, ts) order rather than globally ts-sorted. The
    previous implementation did `.set_index("ts").sort_index()` and then assigned
    the groupby-rolling result back with positional `.values` — but groupby returns
    rows in (account_id, ts) order while the frame was in global-ts order, so the
    per-account values were stamped onto the wrong rows (a silent misalignment that
    corrupted velocity_24h_sum for any account with >1 transaction; it was masked
    for count only because singletons always have count=1). Preserving
    (account_id, ts) order makes the positional assignment correct.
    """
    t = transactions.sort_values(["account_id", "ts"]).reset_index(drop=True)
    sizes = t.groupby("account_id", sort=False)["amount"].transform("size")
    multi = (sizes > 1).to_numpy()

    # Single-transaction accounts trivially have count=1 and sum=their own amount;
    # only the multi-transaction subset needs the (per-group, Python-dispatched)
    # rolling call, which is a huge saving when singletons dominate.
    count_24h = np.ones(len(t), dtype=float)
    sum_24h = t["amount"].to_numpy(dtype=float).copy()
    if multi.any():
        sub = t.loc[multi].set_index("ts")
        grp = sub.groupby("account_id")
        # sub is in (account_id, ts) order; groupby-rolling returns the same order,
        # so positional assignment back onto the multi rows stays aligned.
        c = grp["transaction_id"].rolling("24h").count().reset_index(level=0, drop=True)
        s = grp["amount"].rolling("24h").sum().reset_index(level=0, drop=True)
        count_24h[multi] = c.to_numpy()
        sum_24h[multi] = s.to_numpy()

    t["velocity_24h_count"] = count_24h
    t["velocity_24h_sum"] = sum_24h
    return t


def compute_amount_zscore(transactions: pd.DataFrame) -> pd.Series:
    mean, std = _prior_expanding_mean_std(transactions, "account_id", "amount")
    z = (transactions["amount"] - mean) / std.replace(0, np.nan)
    return z.fillna(0)


def compute_prior_alert_stats(alerts: pd.DataFrame, dispositions: pd.DataFrame) -> pd.DataFrame:
    """
    For each account: how many prior alerts have fired, and what fraction
    of those were false positives (based on seeded dispositions).
    Computed as a running/prior-only stat to avoid leakage into the current alert.
    """
    merged = alerts.merge(dispositions[["alert_id", "decision"]], on="alert_id", how="left")
    merged = merged.sort_values(["account_id", "triggered_at"])

    merged["is_false_positive"] = (merged["decision"] == "not_fraud").astype(int)
    grp = merged.groupby("account_id", sort=False)

    merged["prior_alert_count"] = grp.cumcount()
    # prior_fp_count = running count of FPs BEFORE the current alert. The original
    # used transform(lambda s: s.shift(1).fillna(0).cumsum()) — a per-group Python
    # lambda that dominated enrichment runtime at scale. Equivalent vectorized form:
    # cumulative-including-current minus the current row = sum of strictly-prior rows.
    merged["prior_fp_count"] = grp["is_false_positive"].cumsum() - merged["is_false_positive"]
    merged["prior_fp_rate"] = (merged["prior_fp_count"] / merged["prior_alert_count"].replace(0, np.nan)).fillna(0)

    return merged[["alert_id", "prior_alert_count", "prior_fp_rate"]]


def build_narrative_summary(row: dict) -> str:
    """
    Deterministic, template-based narrative — not LLM-generated, so it stays
    auditable and consistent. Reads the structured facts and produces a
    plain-language summary an analyst can scan in seconds.
    """
    parts = []
    if row.get("amount_zscore", 0) and abs(row["amount_zscore"]) > 3:
        parts.append(f"amount is {row['amount_zscore']:.1f} standard deviations from the account's historical average")
    if row.get("velocity_24h_count", 0) and row["velocity_24h_count"] >= 5:
        parts.append(f"{int(row['velocity_24h_count'])} transactions from this account in the trailing 24 hours")
    if row.get("prior_fp_rate") is not None and row.get("prior_alert_count", 0) >= 3:
        parts.append(f"{row['prior_fp_rate']:.0%} of this account's {int(row['prior_alert_count'])} prior alerts were false positives")
    if row.get("rule_ids"):
        parts.append(f"triggered by: {', '.join(row['rule_ids'])}")

    if not parts:
        return "Low-signal alert with no strongly deviating factors."
    return "This alert was flagged because " + "; ".join(parts) + "."
