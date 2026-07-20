"""
Builds the flat training table for the Phase 2 ranking model.

One row per alert: structured_facts pulled back out of alerts.enrichment
(JSONB, computed in Phase 1) joined with raw transaction/account context
that isn't already in there, plus a one-hot column per rule_id. Label comes
from the seeded dispositions.
"""
import pandas as pd
import numpy as np

CATEGORICAL_COLS = ["transaction_type", "account_type", "card_network", "account_region"]

# Fixed schema written by enrichment/pipeline.py's structured_facts dict.
SF_FIELDS = ["amount", "amount_zscore", "velocity_24h_count", "velocity_24h_sum",
             "prior_alert_count", "prior_fp_rate"]

# Fixed registry so train/test/inference always produce the same columns,
# regardless of which rules actually fired in a given slice of data.
ALL_RULE_IDS = [
    "velocity_10min", "velocity_1h", "amount_zscore", "new_device",
    "new_counterparty_high_amount", "region_mismatch", "round_amount_high",
    "new_account_high_amount", "multi_product_same_day", "missing_identity_high_amount",
]


def load_raw(engine) -> pd.DataFrame:
    """One row per enriched alert: enrichment JSONB + raw transaction/account context + label."""
    query = """
        SELECT
            a.alert_id,
            a.transaction_id,
            a.account_id,
            a.triggered_at,
            a.enrichment,
            t.transaction_type,
            t.counterparty_region,
            t.device_id,
            t.has_identity_data,
            t.amount,
            t.orig_balance_before,
            t.orig_balance_after,
            t.dest_balance_before,
            t.dest_balance_after,
            acc.first_seen_at,
            acc.account_type,
            acc.card_network,
            acc.region_code AS account_region,
            d.decision
        FROM alerts a
        JOIN transactions t ON t.transaction_id = a.transaction_id
        JOIN accounts acc ON acc.account_id = a.account_id
        JOIN dispositions d ON d.alert_id = a.alert_id
        WHERE a.enrichment IS NOT NULL
        ORDER BY a.triggered_at, a.alert_id
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        raise RuntimeError("No enriched alerts found — run rules/engine.py and enrichment/pipeline.py first.")
    return df.reset_index(drop=True)


def _structured_facts_frame(enrichment_col: pd.Series):
    """
    Flatten structured_facts out of each row's enrichment JSONB. Extracts the
    known fixed set of fields directly in a single pass rather than via
    pd.json_normalize — json_normalize is general-purpose (handles arbitrary
    nesting) and measurably slower at millions of rows than a direct .get()
    per field, since the schema here is already known and fixed.
    """
    facts = enrichment_col.apply(lambda e: (e or {}).get("structured_facts") or {})
    rows = [[f.get(field) for field in SF_FIELDS] for f in facts]
    sf = pd.DataFrame(rows, columns=[f"sf_{field}" for field in SF_FIELDS], index=enrichment_col.index)
    rule_ids = facts.apply(lambda f: f.get("rule_ids") or [])
    return sf, rule_ids


def _rule_dummies(rule_ids: pd.Series) -> pd.DataFrame:
    exploded = rule_ids.explode()
    onehot = pd.get_dummies(exploded)
    dummies = onehot.groupby(level=0).max()
    dummies = dummies.reindex(index=rule_ids.index, columns=ALL_RULE_IDS, fill_value=0)
    dummies.columns = [f"rule_{c}" for c in dummies.columns]
    return dummies.astype(int)


def build_feature_matrix(engine):
    """
    Returns (X, y, meta, categorical_cols):
      X                 - flat feature DataFrame, one row per alert
      y                 - binary label Series (1 = confirmed fraud)
      meta              - alert_id / transaction_id / account_id / triggered_at, for
                           the temporal split and the SHAP write-back
      categorical_cols  - column names to mark 'category' dtype for LightGBM
    """
    df = load_raw(engine)

    sf, rule_ids = _structured_facts_frame(df["enrichment"])
    rule_dummies = _rule_dummies(rule_ids)

    df["account_age_days"] = (df["triggered_at"] - df["first_seen_at"]).dt.total_seconds() / 86400
    df["has_device"] = df["device_id"].notna().astype(int)
    df["region_mismatch"] = (df["counterparty_region"].astype(str) != df["account_region"].astype(str)).astype(int)
    df["hour_of_day"] = df["triggered_at"].dt.hour
    df["day_of_week"] = df["triggered_at"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["has_identity_data"] = df["has_identity_data"].astype(int)

    # Balance-derived features — the strongest fraud signal in PaySim (left as NaN,
    # which LightGBM handles natively, for datasets that don't carry balances).
    amount = df["amount"].astype(float)
    ob_before = df["orig_balance_before"].astype(float)
    ob_after = df["orig_balance_after"].astype(float)
    db_before = df["dest_balance_before"].astype(float)
    db_after = df["dest_balance_after"].astype(float)
    df["orig_balance_before"] = ob_before
    df["dest_balance_before"] = db_before
    df["orig_balance_delta"] = ob_after - ob_before
    df["dest_balance_delta"] = db_after - db_before
    # For a legitimate debit/credit the balance arithmetic reconciles to ~0; a
    # non-zero "error" is a classic PaySim fraud tell.
    df["error_balance_orig"] = ob_before - amount - ob_after
    df["error_balance_dest"] = db_after - db_before - amount
    # Account fully drained (balance had funds, went to exactly zero).
    df["orig_emptied"] = ((ob_before > 0) & (ob_after == 0)).astype(int)

    raw_features = df[[
        "transaction_type", "account_type", "card_network", "account_region",
        "has_identity_data", "has_device", "region_mismatch",
        "account_age_days", "hour_of_day", "day_of_week", "is_weekend",
        "orig_balance_before", "dest_balance_before",
        "orig_balance_delta", "dest_balance_delta",
        "error_balance_orig", "error_balance_dest", "orig_emptied",
    ]].copy()

    X = pd.concat([raw_features, sf, rule_dummies], axis=1)
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")

    y = (df["decision"] == "fraud").astype(int)
    meta = df[["alert_id", "transaction_id", "account_id", "triggered_at"]]

    return X, y, meta, CATEGORICAL_COLS
