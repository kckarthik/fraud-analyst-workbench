"""
SHAP reason codes: top-k per-prediction drivers, rendered in plain language.

Lives apart from explain.py so the batch write-back and the online scoring
endpoint produce reason codes through the same code rather than two
implementations that can disagree about ordering, sign, or wording. An analyst
comparing a queued alert against a freshly scored one should not see the same
transaction explained two different ways.
"""
import numpy as np
import pandas as pd

TOP_N_REASONS = 4

# Human-readable gloss for feature names that surface as top SHAP drivers.
FEATURE_DESCRIPTIONS = {
    "sf_amount_zscore": "transaction amount deviates sharply from the account's historical average",
    "sf_velocity_24h_count": "high number of transactions from this account in the trailing 24 hours",
    "sf_velocity_24h_sum": "high total transaction volume from this account in the trailing 24 hours",
    "sf_prior_fp_rate": "this account's prior alerts were mostly false positives",
    "sf_prior_alert_count": "this account has an unusually long alert history",
    "sf_amount": "transaction amount",
    "account_age_days": "account age at time of transaction",
    "has_device": "device presence/absence on this transaction",
    "has_identity_data": "presence/absence of identity signal on this transaction",
    "region_mismatch": "counterparty region differs from the account's home region",
    "orig_balance_before": "origin account balance before the transaction",
    "dest_balance_before": "destination account balance before the transaction",
    "orig_balance_delta": "change in the origin account's balance",
    "dest_balance_delta": "change in the destination account's balance",
    "error_balance_orig": "origin balance doesn't reconcile with the transaction amount",
    "error_balance_dest": "destination balance doesn't reconcile with the transaction amount",
    "orig_emptied": "origin account was fully drained to zero",
    "hour_of_day": "time of day the transaction occurred",
    "day_of_week": "day of week the transaction occurred",
    "is_weekend": "weekend timing",
    "transaction_type": "transaction type",
    "account_type": "account type",
    "card_network": "card network",
    "account_region": "account's home region",
}


def describe(feature: str) -> str:
    if feature in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature]
    if feature.startswith("rule_"):
        return f"rule '{feature[len('rule_'):]}' fired"
    return feature


def json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def compute_reason_codes(model, X: pd.DataFrame, top_n: int = TOP_N_REASONS) -> list:
    import shap  # imported lazily: heavy, and not needed to merely import this module

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Normalize across shap/lightgbm versions to a single (n_rows, n_features)
    # array of contributions toward the positive (fraud) class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    columns = X.columns.to_numpy()
    # Precomputed once: repeated single-cell X.iloc[row, col] lookups inside a
    # tight loop are slow (each goes through pandas' indexing machinery) —
    # plain numpy indexing on a materialized array is much cheaper at scale.
    X_values = X.to_numpy()

    n_rows, n_features = shap_values.shape
    k = min(top_n, n_features)
    abs_vals = np.abs(shap_values)
    # argpartition instead of a full per-row argsort: we only need the top-k
    # by magnitude, not a full ranking, so this is O(n) instead of O(n log n)
    # per row, and done once, vectorized, across the whole matrix.
    top_idx = np.argpartition(-abs_vals, k - 1, axis=1)[:, :k]
    row_range = np.arange(n_rows)[:, None]
    order_within_top = np.argsort(-abs_vals[row_range, top_idx], axis=1)
    top_idx = top_idx[row_range, order_within_top]

    reasons = []
    for row_i in range(n_rows):
        row_shap = shap_values[row_i]
        row_reasons = []
        for feat_i in top_idx[row_i]:
            contribution = float(row_shap[feat_i])
            if contribution == 0:
                continue
            feature = columns[feat_i]
            row_reasons.append({
                "feature": feature,
                "value": json_value(X_values[row_i, feat_i]),
                "shap_value": round(contribution, 4),
                "direction": "increases_risk" if contribution > 0 else "decreases_risk",
                "description": describe(feature),
            })
        reasons.append(row_reasons)
    return reasons
