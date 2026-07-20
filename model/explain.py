"""
SHAP explainability for the Phase 2 model — scores every alert, computes
per-prediction reason codes, and merges them into alerts.enrichment
alongside the Phase 1 structured_facts/narrative_summary.

Requires model/train.py to have been run first (loads its saved model).

Usage:
    python model/explain.py
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "db"))

import joblib
import numpy as np
import pandas as pd
import shap
from db_utils import bulk_jsonb_update, get_engine
from features import build_feature_matrix
from sqlalchemy import text
from tqdm import tqdm

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
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


def _describe(feature: str) -> str:
    if feature in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature]
    if feature.startswith("rule_"):
        return f"rule '{feature[len('rule_'):]}' fired"
    return feature


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def compute_reason_codes(model, X: pd.DataFrame, top_n: int = TOP_N_REASONS) -> list:
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
                "value": _json_value(X_values[row_i, feat_i]),
                "shap_value": round(contribution, 4),
                "direction": "increases_risk" if contribution > 0 else "decreases_risk",
                "description": _describe(feature),
            })
        reasons.append(row_reasons)
    return reasons


def write_back(engine, alert_ids: pd.Series, scores: np.ndarray, reasons: list):
    """
    Bulk merge via COPY-into-temp-table + a single UPDATE ... FROM join
    (see db_utils.bulk_jsonb_update) — ~an order of magnitude faster than
    per-chunk VALUES updates at full scale. Uses merge (||) mode so Phase 1's
    structured_facts/narrative_summary are preserved.
    """
    records = (
        (int(alert_id), json.dumps({"model_score": round(float(score), 4), "reason_codes": reason}))
        # strict=True: these three are parallel arrays from one scoring pass.
        # If they ever diverge in length, zip would silently truncate and stamp
        # scores onto the wrong alerts — the same misalignment class of bug that
        # already cost ~0.15 AUC in the velocity features. Fail loudly instead.
        for alert_id, score, reason in zip(alert_ids, scores, reasons, strict=True)
    )
    bulk_jsonb_update(engine, records, "alerts", "alert_id", "enrichment", mode="merge")


def main(chunk_size: int = 200_000):
    engine = get_engine()
    model_path = os.path.join(ARTIFACT_DIR, "lgbm_model.joblib")
    if not os.path.exists(model_path):
        raise RuntimeError(f"No trained model found at {model_path} — run model/train.py first.")
    model = joblib.load(model_path)

    print("Rebuilding feature matrix for scoring ...")
    X, y, meta, categorical_cols = build_feature_matrix(engine)

    n = len(X)
    print(f"Scoring + explaining {n:,} alerts in chunks of {chunk_size:,} "
          f"(bounds memory and gives visible progress on large runs) ...")
    for start in tqdm(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        X_chunk = X.iloc[start:end]
        meta_chunk = meta.iloc[start:end]

        scores = model.predict_proba(X_chunk)[:, 1]
        reasons = compute_reason_codes(model, X_chunk)
        write_back(engine, meta_chunk["alert_id"], scores, reasons)

    print("Syncing alerts.model_score column (materialized copy of the JSONB value, "
          "for fast indexed ranking — the backend queue query relies on this) ...")
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE alerts SET model_score = (enrichment->>'model_score')::numeric "
            "WHERE enrichment ? 'model_score'"
        ))

    print("Done.")


if __name__ == "__main__":
    main()
