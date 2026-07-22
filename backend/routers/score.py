"""
Real-time scoring: score a transaction as it arrives, rather than in a batch
sweep over a table that already holds it.

The batch path (model/explain.py) and this one deliberately share their model
artifact, their feature derivations (model/features.py), their rules
(rules/rule_definitions.py) and their reason-code rendering
(model/reason_codes.py). Anything computed twice is somewhere the two can
silently diverge.
"""
import os

import joblib
from fastapi import APIRouter, HTTPException

# enrichment_features and reason_codes come via online_features, which already
# loads both by explicit path — importing them again here by name would hit the
# model/features.py vs enrichment/features.py basename collision.
from online_features import (
    UnknownAccountError,
    build_scoring_row,
    enrichment_features,
    reason_codes,
)
from schemas import ReasonCode, ScoreRequest, ScoreResponse

from db import get_app_engine

router = APIRouter(prefix="/api/score", tags=["score"])

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model", "artifacts")
MODEL_PATH = os.path.abspath(os.path.join(ARTIFACT_DIR, "lgbm_model.joblib"))

_model = None


def _get_model():
    """
    Loaded once and held. joblib.load of the LightGBM booster takes long enough
    that doing it per request would dominate the response time, and the model is
    immutable between deploys.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                503, "No trained model available — run model/train.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@router.post("", response_model=ScoreResponse)
def score_transaction(txn: ScoreRequest):
    """Score one transaction against the account's history as of its timestamp."""
    model = _get_model()
    engine = get_app_engine()

    with engine.connect() as conn:
        try:
            X, fired_rule_ids, structured_facts, n_history = build_scoring_row(
                conn, txn.model_dump()
            )
        except UnknownAccountError:
            # A genuinely unknown account is a 422, not a 500: the request is
            # well-formed, it just references something that does not exist. Its
            # history-based features would all be undefined.
            raise HTTPException(
                422, f"Unknown account '{txn.account_id}' — no history to score against."
            ) from None

    score = float(model.predict_proba(X)[:, 1][0])
    reasons = reason_codes.compute_reason_codes(model, X)[0]

    return ScoreResponse(
        account_id=txn.account_id,
        model_score=round(score, 6),
        rule_ids=fired_rule_ids,
        reason_codes=[ReasonCode(**rc) for rc in reasons],
        # Same deterministic template the batch enrichment uses, so a freshly
        # scored transaction reads the same way as one already in the queue.
        narrative_summary=enrichment_features.build_narrative_summary(structured_facts),
        structured_facts=structured_facts,
        history_transactions_used=n_history,
    )
