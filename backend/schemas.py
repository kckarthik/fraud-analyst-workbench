from datetime import datetime
from typing import Any, Optional
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
    model_score: Optional[float] = None
    top_reason: Optional[str] = None
    decision: Optional[str] = None


class AlertListResponse(BaseModel):
    total: int
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
    counterparty_id: Optional[str] = None
    account_type: Optional[str] = None
    card_network: Optional[str] = None
    region_code: Optional[str] = None
    model_score: Optional[float] = None
    reason_codes: list[ReasonCode] = []
    narrative_summary: Optional[str] = None
    structured_facts: dict = {}
    decision: Optional[str] = None
    notes: Optional[str] = None


class DispositionCreate(BaseModel):
    decision: str
    analyst_id: str = "web_analyst"
    notes: Optional[str] = None


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sql: Optional[str] = None
    columns: Optional[list[str]] = None
    rows: Optional[list[dict]] = None
    error: Optional[str] = None
