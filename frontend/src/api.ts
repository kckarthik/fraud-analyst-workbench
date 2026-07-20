import type { AlertListResponse, AlertDetail, ChatResponse } from './types';

const BASE = '/api';

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export function fetchAlerts(params: {
  limit?: number;
  offset?: number;
  minScore?: number;
  status?: string;
}): Promise<AlertListResponse> {
  const q = new URLSearchParams();
  if (params.limit) q.set('limit', String(params.limit));
  if (params.offset) q.set('offset', String(params.offset));
  if (params.minScore !== undefined) q.set('min_score', String(params.minScore));
  if (params.status) q.set('status', params.status);
  return fetch(`${BASE}/alerts?${q}`).then((r) => handle<AlertListResponse>(r));
}

export function fetchAlertDetail(id: number): Promise<AlertDetail> {
  return fetch(`${BASE}/alerts/${id}`).then((r) => handle<AlertDetail>(r));
}

export function submitDisposition(
  id: number,
  decision: 'fraud' | 'not_fraud',
  notes?: string,
): Promise<{ ok: boolean; disposition_id: number }> {
  return fetch(`${BASE}/alerts/${id}/disposition`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision, notes, analyst_id: 'web_analyst' }),
  }).then((r) => handle(r));
}

export function askAgent(question: string): Promise<ChatResponse> {
  return fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  }).then((r) => handle<ChatResponse>(r));
}
