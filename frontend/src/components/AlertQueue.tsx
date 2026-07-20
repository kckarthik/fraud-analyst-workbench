import { useEffect, useState } from 'react';
import { fetchAlerts } from '../api';
import type { AlertListItem } from '../types';

const PAGE_SIZE = 25;
const STATUS_OPTIONS = [
  { value: '', label: 'All' },
  { value: 'open', label: 'Open' },
  { value: 'closed', label: 'Closed' },
];

interface Props {
  selectedId: number | null;
  onSelect: (id: number) => void;
  refreshKey: number;
}

export default function AlertQueue({ selectedId, onSelect, refreshKey }: Props) {
  const [items, setItems] = useState<AlertListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [statusFilter, setStatusFilter] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchAlerts({ limit: PAGE_SIZE, offset: page * PAGE_SIZE, status: statusFilter || undefined })
      .then((res) => {
        setItems(res.items);
        setTotal(res.total);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [page, statusFilter, refreshKey]);

  const maxPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);

  return (
    <div className="panel queue-panel">
      <div className="panel-toolbar">
        <div className="toolbar-left">
          <span className="count-chip">{total.toLocaleString()}</span>
          <span className="toolbar-label">ranked alerts</span>
        </div>
        <div className="segmented">
          {STATUS_OPTIONS.map((o) => (
            <button
              key={o.value}
              className={statusFilter === o.value ? 'active' : ''}
              onClick={() => {
                setStatusFilter(o.value);
                setPage(0);
              }}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th className="col-risk">Risk</th>
              <th>Alert</th>
              <th className="num">Amount</th>
              <th>Top reason</th>
              <th className="col-state">State</th>
            </tr>
          </thead>
          <tbody>
            {items.map((a) => (
              <tr
                key={a.alert_id}
                className={a.alert_id === selectedId ? 'selected' : ''}
                onClick={() => onSelect(a.alert_id)}
              >
                <td className="col-risk">
                  <RiskMeter score={a.model_score} />
                </td>
                <td>
                  <div className="cell-primary mono">#{a.alert_id}</div>
                  <div className="cell-secondary">{a.transaction_type}</div>
                </td>
                <td className="num">
                  <span className="amount tabular">${a.amount.toLocaleString()}</span>
                </td>
                <td>
                  <span className="top-reason">{a.top_reason ?? '—'}</span>
                </td>
                <td className="col-state">
                  {a.decision ? (
                    <span className={`badge decision-${a.decision}`}>
                      {a.decision === 'fraud' ? 'Fraud' : 'Cleared'}
                    </span>
                  ) : (
                    <span className={`badge status-${a.status}`}>{a.status}</span>
                  )}
                </td>
              </tr>
            ))}
            {!loading && items.length === 0 && (
              <tr>
                <td colSpan={5} className="empty-row">
                  No alerts match this filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
        {loading && <div className="loading-overlay">Loading…</div>}
      </div>

      <div className="panel-footer">
        <span className="footer-info">
          Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} of {total.toLocaleString()}
        </span>
        <div className="pager">
          <button disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
            ←
          </button>
          <span className="pager-label">
            {page + 1} / {maxPage + 1}
          </span>
          <button disabled={page >= maxPage} onClick={() => setPage((p) => p + 1)}>
            →
          </button>
        </div>
      </div>
    </div>
  );
}

function RiskMeter({ score }: { score: number | null }) {
  if (score === null) return <span className="risk-na">—</span>;
  const pct = Math.round(score * 100);
  const level = score >= 0.9 ? 'high' : score >= 0.5 ? 'mid' : 'low';
  return (
    <div className={`risk risk-${level}`} title={String(score)}>
      <div className="risk-track">
        <span className="risk-fill" style={{ width: `${Math.max(pct, 3)}%` }} />
      </div>
      <span className="risk-val tabular">{pct}</span>
    </div>
  );
}
