import { useEffect, useState } from 'react';
import { fetchAlertDetail, submitDisposition } from '../api';
import type { AlertDetail } from '../types';

interface Props {
  alertId: number | null;
  onDispositionSaved: () => void;
}

export default function AlertDetailPanel({ alertId, onDispositionSaved }: Props) {
  const [detail, setDetail] = useState<AlertDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (alertId === null) {
      setDetail(null);
      return;
    }
    setLoading(true);
    setError(null);
    setNotes('');
    fetchAlertDetail(alertId)
      .then(setDetail)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [alertId]);

  if (alertId === null) {
    return (
      <div className="panel detail-panel">
        <div className="detail-empty">
          <div className="detail-empty-icon" aria-hidden>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.3">
              <path d="M4 5h16M4 12h16M4 19h10" strokeLinecap="round" />
            </svg>
          </div>
          <p>Select an alert to inspect its risk score, model reason codes, and transaction facts.</p>
        </div>
      </div>
    );
  }

  if (loading) return <div className="panel detail-panel"><div className="detail-loading">Loading…</div></div>;
  if (error) return <div className="panel detail-panel"><div className="error-banner">{error}</div></div>;
  if (!detail) return null;

  async function decide(decision: 'fraud' | 'not_fraud') {
    setSaving(true);
    try {
      await submitDisposition(detail!.alert_id, decision, notes || undefined);
      const refreshed = await fetchAlertDetail(detail!.alert_id);
      setDetail(refreshed);
      onDispositionSaved();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }

  const pct = detail.model_score !== null ? Math.round(detail.model_score * 100) : null;
  const level = detail.model_score === null ? 'na' : detail.model_score >= 0.9 ? 'high' : detail.model_score >= 0.5 ? 'mid' : 'low';
  const maxAbs = Math.max(1e-9, ...detail.reason_codes.map((r) => Math.abs(r.shap_value)));

  return (
    <div className="panel detail-panel">
      <div className="detail-scroll">
        <div className={`detail-hero risk-${level}`}>
          <div className="hero-left">
            <div className="hero-eyebrow mono">ALERT #{detail.alert_id}</div>
            <div className="hero-amount tabular">${detail.amount.toLocaleString()}</div>
            <div className="hero-type">{detail.transaction_type}</div>
          </div>
          {pct !== null && (
            <div className="hero-score">
              <div className="hero-score-val tabular">{pct}</div>
              <div className="hero-score-label">risk score</div>
            </div>
          )}
        </div>

        <p className="narrative">{detail.narrative_summary}</p>

        <section className="detail-section">
          <h3 className="section-title">Model reasoning (SHAP)</h3>
          <ul className="reason-list">
            {detail.reason_codes.map((rc, i) => {
              const w = (Math.abs(rc.shap_value) / maxAbs) * 100;
              return (
                <li key={i} className={`reason reason-${rc.direction}`}>
                  <div className="reason-head">
                    <span className="reason-desc">{rc.description}</span>
                    <span className="reason-shap tabular">
                      {rc.direction === 'increases_risk' ? '+' : ''}
                      {rc.shap_value.toFixed(2)}
                    </span>
                  </div>
                  <div className="reason-track">
                    <span className="reason-fill" style={{ width: `${Math.max(w, 4)}%` }} />
                  </div>
                  <div className="reason-value">value: {String(rc.value)}</div>
                </li>
              );
            })}
          </ul>
        </section>

        <section className="detail-section">
          <h3 className="section-title">Transaction</h3>
          <div className="fact-grid">
            <Fact label="Account" value={detail.account_id} mono />
            <Fact label="Account type" value={detail.account_type ?? '—'} />
            <Fact label="Counterparty" value={detail.counterparty_id ?? '—'} mono />
            <Fact label="Card network" value={detail.card_network ?? '—'} />
            <Fact label="Triggered rules" value={detail.rule_ids.join(', ') || '—'} full />
          </div>
        </section>

        <details className="raw-facts">
          <summary>Raw structured facts (JSON)</summary>
          <pre className="mono">{JSON.stringify(detail.structured_facts, null, 2)}</pre>
        </details>
      </div>

      <div className="detail-actions">
        {detail.decision && (
          <div className="current-decision">
            Current disposition:{' '}
            <span className={`badge decision-${detail.decision}`}>
              {detail.decision === 'fraud' ? 'Fraud' : 'Cleared'}
            </span>
          </div>
        )}
        <textarea placeholder="Add a note (optional)…" value={notes} onChange={(e) => setNotes(e.target.value)} />
        <div className="action-buttons">
          <button className="btn btn-fraud" disabled={saving} onClick={() => decide('fraud')}>
            Confirm Fraud
          </button>
          <button className="btn btn-clear" disabled={saving} onClick={() => decide('not_fraud')}>
            Mark Cleared
          </button>
        </div>
      </div>
    </div>
  );
}

function Fact({ label, value, mono, full }: { label: string; value: string; mono?: boolean; full?: boolean }) {
  return (
    <div className={`fact ${full ? 'fact-full' : ''}`}>
      <span className="fact-label">{label}</span>
      <span className={`fact-value ${mono ? 'mono' : ''}`}>{value}</span>
    </div>
  );
}
