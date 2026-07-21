import json

from fastapi import APIRouter, HTTPException, Query
from schemas import AlertDetail, AlertListResponse, DispositionCreate, ReasonCode
from sqlalchemy import text

from db import get_app_engine

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

# rules/engine.py backfills a disposition for every alert from the dataset's
# ground-truth is_fraud label, under this analyst_id. Those rows are the training
# label and must stay in the database — but they are the answer key, so the
# workbench must not surface them as if an analyst had reached that decision.
# Showing them would let the queue display "Fraud" next to an alert nobody has
# reviewed, and would let an analyst read the label instead of investigating.
#
# Filtered with IS DISTINCT FROM rather than <>: analyst_id is nullable, and a
# plain <> would evaluate to NULL for a row with no analyst recorded, silently
# hiding a genuine human disposition. Unknown analyst still means "not seeded".
SEED_ANALYST_ID = "seed_ground_truth"

# Below this many estimated rows an exact COUNT(*) is cheap enough to just run.
COUNT_EXACT_THRESHOLD = 50_000


def _queue_total(conn, where_sql: str, params: dict) -> tuple[int, bool]:
    """
    Row count for the pagination total, as (count, is_estimate).

    An exact COUNT(*) over ~1M alerts is an unindexed scan that dominated this
    endpoint — the ranked query itself runs in 2.7ms, and the count made the
    request ~0.6s — purely to render "of N" in the footer. So ask the planner
    for its row estimate first: if the result is small the exact count is cheap
    and we run it (filtered views stay precise), and if it's large we show the
    estimate, which is the only part of the response nobody needs to be exact.
    """
    plan = conn.execute(
        text(f"EXPLAIN (FORMAT JSON) SELECT 1 FROM alerts a WHERE {where_sql}"), params
    ).scalar()
    # psycopg2 hands back EXPLAIN FORMAT JSON as text on some versions and as
    # parsed JSON on others.
    if isinstance(plan, str):
        plan = json.loads(plan)
    estimated = int(plan[0]["Plan"]["Plan Rows"])

    if estimated < COUNT_EXACT_THRESHOLD:
        exact = conn.execute(text(f"SELECT COUNT(*) FROM alerts a WHERE {where_sql}"), params).scalar()
        return int(exact), False
    return estimated, True


@router.get("", response_model=AlertListResponse)
def list_alerts(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = None,
    min_score: float = 0.0,
):
    """Ranked alert queue: highest fraud-risk score first."""
    engine = get_app_engine()
    where = ["a.model_score IS NOT NULL", "a.model_score >= :min_score"]
    where_params: dict = {"min_score": min_score}
    if status:
        where.append("a.status = :status")
        where_params["status"] = status
    where_sql = " AND ".join(where)

    with engine.connect() as conn:
        total, total_is_estimate = _queue_total(conn, where_sql, where_params)
        rows = conn.execute(
            text(
                f"""
                SELECT a.alert_id, a.transaction_id, a.account_id, a.triggered_at, a.status,
                       a.rule_ids, t.amount, t.transaction_type,
                       a.model_score,
                       a.enrichment->'reason_codes'->0->>'description' AS top_reason,
                       d.decision
                FROM alerts a
                JOIN transactions t ON t.transaction_id = a.transaction_id
                LEFT JOIN LATERAL (
                    SELECT decision FROM dispositions
                    WHERE alert_id = a.alert_id AND analyst_id IS DISTINCT FROM :seed_analyst
                    ORDER BY decided_at DESC LIMIT 1
                ) d ON true
                WHERE {where_sql}
                ORDER BY a.model_score DESC NULLS LAST
                LIMIT :limit OFFSET :offset
                """
            ),
            {**where_params, "seed_analyst": SEED_ANALYST_ID, "limit": limit, "offset": offset},
        ).mappings().all()

    return {"total": total, "total_is_estimate": total_is_estimate, "items": [dict(r) for r in rows]}


@router.get("/{alert_id}", response_model=AlertDetail)
def get_alert(alert_id: int):
    engine = get_app_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT a.alert_id, a.transaction_id, a.account_id, a.triggered_at, a.status, a.rule_ids,
                       a.enrichment, t.amount, t.transaction_type, t.counterparty_id,
                       acc.account_type, acc.card_network, acc.region_code,
                       d.decision, d.notes
                FROM alerts a
                JOIN transactions t ON t.transaction_id = a.transaction_id
                JOIN accounts acc ON acc.account_id = a.account_id
                LEFT JOIN LATERAL (
                    SELECT decision, notes FROM dispositions
                    WHERE alert_id = a.alert_id AND analyst_id IS DISTINCT FROM :seed_analyst
                    ORDER BY decided_at DESC LIMIT 1
                ) d ON true
                WHERE a.alert_id = :aid
                """
            ),
            {"aid": alert_id, "seed_analyst": SEED_ANALYST_ID},
        ).mappings().first()

    if not row:
        raise HTTPException(404, "Alert not found")

    enrichment = row["enrichment"] or {}
    return AlertDetail(
        alert_id=row["alert_id"],
        transaction_id=row["transaction_id"],
        account_id=row["account_id"],
        triggered_at=row["triggered_at"],
        status=row["status"],
        rule_ids=row["rule_ids"] or [],
        amount=float(row["amount"]),
        transaction_type=row["transaction_type"],
        counterparty_id=row["counterparty_id"],
        account_type=row["account_type"],
        card_network=row["card_network"],
        region_code=row["region_code"],
        model_score=enrichment.get("model_score"),
        reason_codes=[ReasonCode(**rc) for rc in enrichment.get("reason_codes", [])],
        narrative_summary=enrichment.get("narrative_summary"),
        structured_facts=enrichment.get("structured_facts", {}),
        decision=row["decision"],
        notes=row["notes"],
    )


@router.post("/{alert_id}/disposition")
def create_disposition(alert_id: int, body: DispositionCreate):
    if body.decision not in ("fraud", "not_fraud"):
        raise HTTPException(400, "decision must be 'fraud' or 'not_fraud'")

    engine = get_app_engine()
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT 1 FROM alerts WHERE alert_id = :aid"), {"aid": alert_id}).first()
        if not exists:
            raise HTTPException(404, "Alert not found")

        result = conn.execute(
            text(
                """
                INSERT INTO dispositions (alert_id, analyst_id, decision, notes)
                VALUES (:aid, :analyst_id, :decision, :notes)
                RETURNING disposition_id
                """
            ),
            {"aid": alert_id, "analyst_id": body.analyst_id, "decision": body.decision, "notes": body.notes},
        )
        disposition_id = result.scalar()
        conn.execute(text("UPDATE alerts SET status = 'closed' WHERE alert_id = :aid"), {"aid": alert_id})

    return {"ok": True, "disposition_id": disposition_id}
