"""
Rule definitions for the synthetic alert-generation engine.

Each rule is a function that takes the full transactions DataFrame
(sorted by account_id, ts) plus the accounts DataFrame, and returns a
boolean Series aligned to the transactions index: True where the rule fires.

These are simplified analogues of real bank fraud rules — not a claim
to replicate proprietary rule logic, but structurally representative of
the kind of velocity/behavioral/deviation checks banks actually run.
"""
import numpy as np
import pandas as pd


def prior_expanding_stats(df: pd.DataFrame, group_col: str, val_col: str):
    """
    Vectorized prior-only (shift-1) expanding count/mean/std per group, computed
    via cumulative sums instead of groupby.transform(lambda s: s.shift(1).expanding()...).

    The per-group Python-lambda form is O(n) *per group* with heavy interpreter
    overhead and becomes the dominant cost at millions of accounts; this computes
    the identical statistics with a handful of vectorized C-level cumsum passes.
    Returns (prior_count, prior_mean, prior_std) as float Series aligned to df,
    matching pandas' expanding semantics: mean is NaN with 0 prior points, std
    (sample, ddof=1) is NaN with fewer than 2 prior points.

    Assumes df is already sorted by (group_col, <time>) so within-group cumulative
    order is chronological — the same ordering the callers rely on.
    """
    vals = df[val_col].astype(float)
    g = vals.groupby(df[group_col], sort=False)
    # cumulative INCLUDING current row, then subtract current to get prior-only
    prior_count = g.cumcount().astype(float)                 # rows strictly before current
    prior_sum = g.cumsum() - vals
    prior_sumsq = (vals * vals).groupby(df[group_col], sort=False).cumsum() - vals * vals

    n = prior_count.replace(0, np.nan)
    prior_mean = prior_sum / n
    # sample variance: (Σx² - (Σx)²/n) / (n-1), only defined for n >= 2
    denom = (prior_count - 1).where(prior_count >= 2, np.nan)
    var = (prior_sumsq - (prior_sum * prior_sum) / n) / denom
    prior_std = np.sqrt(var.clip(lower=0))                   # clip guards tiny negative float error
    return prior_count, prior_mean, prior_std


def _rolling_window_count(df: pd.DataFrame, window: str) -> np.ndarray:
    """
    Trailing time-window transaction count per account, aligned to df order.

    groupby(...).rolling(window) dispatches per group in Python and is slow when
    there are millions of (mostly single-transaction) accounts. A single-transaction
    account always has a rolling count of 1, so we shortcut those and run the real
    rolling only on the multi-transaction subset — identical result, a fraction of
    the work whenever singletons dominate.
    """
    sizes = df.groupby("account_id", sort=False)["ts"].transform("size")
    multi = (sizes > 1).to_numpy()
    result = np.ones(len(df), dtype=float)
    if multi.any():
        sub = df.loc[multi]
        c = sub.set_index("ts").groupby("account_id")["transaction_id"].rolling(window).count()
        result[multi] = c.reset_index(level=0, drop=True).to_numpy()
    return result


def rule_velocity_10min(df: pd.DataFrame, min_count: int = 3) -> pd.Series:
    """3+ transactions from the same account within a trailing 10-minute window."""
    return _rolling_window_count(df, "10min") >= min_count


def rule_velocity_1h(df: pd.DataFrame, min_count: int = 5) -> pd.Series:
    """5+ transactions from the same account within a trailing 1-hour window."""
    return _rolling_window_count(df, "1h") >= min_count


def rule_amount_zscore(df: pd.DataFrame, z_thresh: float = 3.0, min_history: int = 5) -> pd.Series:
    """
    Transaction amount deviates > z_thresh standard deviations from the
    account's own historical (prior-only) mean. Requires min_history prior
    transactions to avoid flagging every early transaction on a new account.
    """
    prior_count, prior_mean, prior_std = prior_expanding_stats(df, "account_id", "amount")
    z = (df["amount"] - prior_mean) / prior_std.replace(0, np.nan)
    fires = (z.abs() > z_thresh) & (prior_count >= min_history)
    return fires.fillna(False).values


def rule_new_device(df: pd.DataFrame) -> pd.Series:
    """Device not previously seen on this account (and device_id is known)."""
    has_device = df["device_id"].notna()
    seen_before = df.groupby(["account_id", "device_id"]).cumcount() > 0
    return (has_device & ~seen_before).values


def rule_new_counterparty_high_amount(df: pd.DataFrame, multiplier: float = 2.0, min_history: int = 3) -> pd.Series:
    """First transaction with this counterparty AND amount > multiplier x account's historical average."""
    seen_before = df.groupby(["account_id", "counterparty_id"]).cumcount() > 0
    prior_count, hist_mean, _ = prior_expanding_stats(df, "account_id", "amount")
    high_amount = df["amount"] > (multiplier * hist_mean)
    fires = (~seen_before) & high_amount.fillna(False) & (prior_count >= min_history)
    return fires.fillna(False).values


def rule_region_mismatch(df: pd.DataFrame, accounts: pd.DataFrame) -> pd.Series:
    """Transaction counterparty region differs from the account's home region."""
    acct_region = accounts.set_index("account_id")["region_code"]
    home_region = df["account_id"].map(acct_region)
    return (df["counterparty_region"].astype(str) != home_region.astype(str)).values


def rule_round_amount_high(df: pd.DataFrame, threshold: float = 500) -> pd.Series:
    """Suspiciously round amount (multiple of 100) above a materiality threshold."""
    return ((df["amount"] >= threshold) & (df["amount"] % 100 == 0)).values


def rule_new_account_high_amount(df: pd.DataFrame, accounts: pd.DataFrame,
                                    days: int = 1, threshold: float = 500) -> pd.Series:
    """Large transaction within `days` of the account's first-seen transaction."""
    first_seen = df["account_id"].map(accounts.set_index("account_id")["first_seen_at"])
    age = (df["ts"] - first_seen).dt.total_seconds() / 86400
    return ((age <= days) & (df["amount"] >= threshold)).values


def rule_multi_product_same_day(df: pd.DataFrame) -> pd.Series:
    """
    Account has used more than one distinct transaction type so far today —
    counting only this transaction and the ones before it.

    The original computed `transform("nunique")` over the whole (account, day)
    group, which spans transactions that had not happened yet. Offline that
    looked fine; every row of the day saw the day's full set of product types.
    It is not point-in-time correct, and the divergence is not academic: a rule
    that needs to know what the account will do at 3pm cannot be evaluated on a
    transaction arriving at 10am, so it could not be reproduced by the online
    scoring path at all. Found by scoring already-scored transactions through
    POST /api/score and diffing the fired rules — 52 of 200 disagreed, every one
    of them on the account's first transaction of the day.

    Prior-only formulation: distinct types among rows 0..i exceeds one exactly
    when some row at or before i differs from the day's first type, so a running
    maximum over that comparison gives the expanding distinct-count test without
    a per-group Python pass.
    """
    day = df["ts"].dt.date
    grouped = df.groupby(["account_id", day], sort=False)["transaction_type"]
    differs_from_first = df["transaction_type"] != grouped.transform("first")
    return differs_from_first.groupby([df["account_id"], day], sort=False).cummax().values


def rule_missing_identity_high_amount(df: pd.DataFrame, threshold: float = 500) -> pd.Series:
    """No device/identity signal available AND transaction amount is high."""
    return ((~df["has_identity_data"]) & (df["amount"] >= threshold)).values


# Registry: rule_id -> (function, description, kwargs)
RULE_REGISTRY = {
    "velocity_10min": (rule_velocity_10min, "3+ transactions within a trailing 10-minute window", {}),
    "velocity_1h": (rule_velocity_1h, "5+ transactions within a trailing 1-hour window", {}),
    "amount_zscore": (rule_amount_zscore, "Amount > 3 std devs from account's historical mean", {}),
    "new_device": (rule_new_device, "First transaction from a device not previously seen on this account", {}),
    "new_counterparty_high_amount": (rule_new_counterparty_high_amount,
                                       "First transaction with counterparty + amount > 2x account average", {}),
    "region_mismatch": (rule_region_mismatch, "Counterparty region differs from account's home region", {"needs_accounts": True}),
    "round_amount_high": (rule_round_amount_high, "Round-dollar amount >= $500", {}),
    "new_account_high_amount": (rule_new_account_high_amount,
                                  "High-value transaction within 1 day of account first seen", {"needs_accounts": True}),
    "multi_product_same_day": (rule_multi_product_same_day, "More than one transaction type used same day", {}),
    "missing_identity_high_amount": (rule_missing_identity_high_amount,
                                       "No device/identity signal + high transaction amount", {}),
}
