import { useState } from 'react';
import AlertQueue from './components/AlertQueue';
import AlertDetailPanel from './components/AlertDetailPanel';
import ChatPanel from './components/ChatPanel';
import './App.css';

type Tab = 'queue' | 'chat';

export default function App() {
  const [tab, setTab] = useState<Tab>('queue');
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 2 4 5v6c0 5 3.5 8.5 8 11 4.5-2.5 8-6 8-11V5l-8-3Z"
                fill="currentColor"
                opacity="0.25"
              />
              <path
                d="M12 2 4 5v6c0 5 3.5 8.5 8 11 4.5-2.5 8-6 8-11V5l-8-3Z"
                stroke="currentColor"
                strokeWidth="1.5"
              />
              <path d="m9 12 2 2 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div className="brand-text">
            <span className="brand-name">Fraud Intel</span>
            <span className="brand-sub">Analyst Workbench</span>
          </div>
        </div>

        <nav className="nav">
          <button className={`nav-item ${tab === 'queue' ? 'active' : ''}`} onClick={() => setTab('queue')}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M4 6h16M4 12h16M4 18h10" strokeLinecap="round" />
            </svg>
            Alert Queue
          </button>
          <button className={`nav-item ${tab === 'chat' ? 'active' : ''}`} onClick={() => setTab('chat')}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M21 12a8 8 0 0 1-11.3 7.3L4 21l1.7-5.7A8 8 0 1 1 21 12Z" strokeLinejoin="round" />
            </svg>
            Ask the Data
          </button>
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot" />
          <div>
            <div className="sf-title">System live</div>
            <div className="sf-sub">LightGBM · SHAP · local LLM</div>
          </div>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="topbar-title">
            <h1>{tab === 'queue' ? 'Alert Queue' : 'Ask the Data'}</h1>
            <span className="topbar-sub">
              {tab === 'queue'
                ? 'Alerts ranked by model fraud-risk score, highest first'
                : 'Natural-language questions answered over the alert database'}
            </span>
          </div>
        </header>

        <div className="content">
          {tab === 'queue' ? (
            <div className="queue-layout">
              <AlertQueue selectedId={selectedId} onSelect={setSelectedId} refreshKey={refreshKey} />
              <AlertDetailPanel alertId={selectedId} onDispositionSaved={() => setRefreshKey((k) => k + 1)} />
            </div>
          ) : (
            <ChatPanel />
          )}
        </div>
      </div>
    </div>
  );
}
