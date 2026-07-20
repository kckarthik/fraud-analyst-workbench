export interface ReasonCode {
  feature: string;
  value: unknown;
  direction: 'increases_risk' | 'decreases_risk';
  shap_value: number;
  description: string;
}

export interface AlertListItem {
  alert_id: number;
  transaction_id: string;
  account_id: string;
  triggered_at: string;
  status: string;
  rule_ids: string[];
  amount: number;
  transaction_type: string;
  model_score: number | null;
  top_reason: string | null;
  decision: string | null;
}

export interface AlertListResponse {
  total: number;
  items: AlertListItem[];
}

export interface AlertDetail {
  alert_id: number;
  transaction_id: string;
  account_id: string;
  triggered_at: string;
  status: string;
  rule_ids: string[];
  amount: number;
  transaction_type: string;
  counterparty_id: string | null;
  account_type: string | null;
  card_network: string | null;
  region_code: string | null;
  model_score: number | null;
  reason_codes: ReasonCode[];
  narrative_summary: string | null;
  structured_facts: Record<string, unknown>;
  decision: string | null;
  notes: string | null;
}

export interface ChatResponse {
  answer: string;
  sql: string | null;
  columns: string[] | null;
  rows: Record<string, unknown>[] | null;
  error: string | null;
}

export interface ChatTurn {
  question: string;
  response?: ChatResponse;
  loading?: boolean;
}
