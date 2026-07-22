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
from db_utils import bulk_jsonb_update, get_engine
from features import build_feature_matrix
from reason_codes import compute_reason_codes
from sqlalchemy import text
from tqdm import tqdm

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

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

    # The UPDATE above just rewrote model_score on every row, and until the
    # planner re-samples the table its statistics still describe the column as
    # entirely NULL. That makes it estimate ~0 rows for the queue's
    # "model_score IS NOT NULL" predicate, which silently defeats the estimated
    # count in routers/alerts.py — the endpoint falls back to an exact COUNT(*)
    # over ~1M rows precisely when the table is at its largest. Autovacuum fixes
    # this eventually; ANALYZE fixes it now, and costs seconds.
    print("Refreshing planner statistics on alerts (ANALYZE) ...")
    with engine.begin() as conn:
        conn.execute(text("ANALYZE alerts"))

    print("Done.")


if __name__ == "__main__":
    main()
