import { useState } from 'react';
import { askAgent } from '../api';
import type { ChatTurn } from '../types';

const SUGGESTIONS = [
  'How many confirmed fraud alerts are there?',
  'What transaction types appear most in high-risk alerts?',
  'Average amount of alerts confirmed as fraud',
];

export default function ChatPanel() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [question, setQuestion] = useState('');

  async function send(q?: string) {
    const text = (q ?? question).trim();
    if (!text) return;
    setQuestion('');
    setTurns((t) => [...t, { question: text, loading: true }]);
    try {
      const response = await askAgent(text);
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, response, loading: false } : turn)));
    } catch (e) {
      setTurns((t) =>
        t.map((turn, i) =>
          i === t.length - 1
            ? { ...turn, loading: false, response: { answer: '', sql: null, columns: null, rows: null, error: String(e) } }
            : turn,
        ),
      );
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-scroll">
        {turns.length === 0 && (
          <div className="chat-welcome">
            <h2>Ask a question about the alert data</h2>
            <p className="chat-welcome-sub">
              A local LangGraph agent (Ollama) generates SQL against a read-only database role, validates it,
              runs it, and summarizes the result. Runs on CPU — answers take ~30–60s.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="suggestion" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn, i) => (
          <div key={i} className="chat-turn">
            <div className="chat-question">
              <span className="chat-avatar you">You</span>
              <div className="chat-bubble">{turn.question}</div>
            </div>
            <div className="chat-answer">
              <span className="chat-avatar bot" aria-hidden>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M21 12a8 8 0 0 1-11.3 7.3L4 21l1.7-5.7A8 8 0 1 1 21 12Z" strokeLinejoin="round" />
                </svg>
              </span>
              <div className="chat-bubble bot-bubble">
                {turn.loading ? (
                  <span className="thinking">
                    <span></span><span></span><span></span>
                  </span>
                ) : turn.response?.error && !turn.response.sql ? (
                  <div className="error-inline">{turn.response.error}</div>
                ) : (
                  <>
                    <p>{turn.response?.answer}</p>
                    {turn.response?.sql && (
                      <details className="sql-details">
                        <summary>SQL & result</summary>
                        <pre className="mono">{turn.response.sql}</pre>
                        {turn.response.rows && (
                          <pre className="mono result-json">{JSON.stringify(turn.response.rows.slice(0, 10), null, 2)}</pre>
                        )}
                      </details>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="chat-composer">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder="Ask about alerts, fraud, transactions…"
        />
        <button onClick={() => send()} disabled={!question.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
