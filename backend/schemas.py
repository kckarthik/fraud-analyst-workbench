from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ReasonCode(BaseModel):
    feature: str
    value: Any
    direction: str
    shap_value: float
    description: str


class AlertListItem(BaseModel):
    alert_id: int
    transaction_id: str
    account_id: str
    triggered_at: datetime
    status: str
    rule_ids: list[str]
    amount: float
    transaction_type: str
    model_score: float | None = None
    top_reason: str | None = None
    decision: str | None = None


class AlertListResponse(BaseModel):
    total: int
    # True when `total` is the planner's row estimate rather than an exact count
    # (see routers/alerts.py:_queue_total). The UI renders it as "~988,600" so a
    # figure that can drift by a percent or two is never shown as though it were
    # authoritative.
    total_is_estimate: bool = False
    items: list[AlertListItem]


class AlertDetail(BaseModel):
    alert_id: int
    transaction_id: str
    account_id: str
    triggered_at: datetime
    status: str
    rule_ids: list[str]
    amount: float
    transaction_type: str
    counterparty_id: str | None = None
    account_type: str | None = None
    card_network: str | None = None
    region_code: str | None = None
    model_score: float | None = None
    reason_codes: list[ReasonCode] = []
    narrative_summary: str | None = None
    structured_facts: dict = {}
    decision: str | None = None
    notes: str | None = None


class DispositionCreate(BaseModel):
    decision: str
    analyst_id: str = "web_analyst"
    notes: str | None = None


class ScoreRequest(BaseModel):
    """
    A transaction to score before it is written anywhere. Field names mirror the
    transactions table so a caller shaping a payload has one vocabulary, not two.
    """
    account_id: str
    ts: datetime
    amount: float
    transaction_type: str
    counterparty_id: str | None = None
    counterparty_region: str | None = None
    device_id: str | None = None
    has_identity_data: bool = False
    orig_balance_before: float | None = None
    orig_balance_after: float | None = None
    dest_balance_before: float | None = None
    dest_balance_after: float | None = None


class ScoreResponse(BaseModel):
    account_id: str
    model_score: float
    # Rules that fired for this transaction, evaluated against the account's
    # history exactly as the batch engine would have.
    rule_ids: list[str] = []
    reason_codes: list[ReasonCode] = []
    narrative_summary: str | None = None
    structured_facts: dict = {}
    # How much account history the score was computed against. A score built on
    # two prior transactions is not the same claim as one built on two hundred,
    # and the caller cannot tell from the number alone.
    history_transactions_used: int = 0


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sql: str | None = None
    columns: list[str] | None = None
    rows: list[dict] | None = None
    error: str | None = None
