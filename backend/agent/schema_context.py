"""
Compact schema description fed to the SQL-generation prompt. Deliberately
scoped to exactly what the read-only role can see (see db/setup_readonly_role.sql):
analyst_transactions (not the raw transactions table, which hides is_fraud).
"""

SCHEMA_CONTEXT = """
Tables/views you may query (Postgres):

accounts(account_id TEXT, first_seen_at TIMESTAMP, account_type TEXT, card_network TEXT, region_code TEXT)

analyst_transactions(transaction_id TEXT, account_id TEXT, ts TIMESTAMP, amount NUMERIC,
  currency TEXT, transaction_type TEXT, counterparty_id TEXT, counterparty_region TEXT,
  device_id TEXT, ip_proxy TEXT, has_identity_data BOOLEAN,
  orig_balance_before NUMERIC, orig_balance_after NUMERIC,
  dest_balance_before NUMERIC, dest_balance_after NUMERIC)
  -- transaction_type is one of: PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT

alerts(alert_id INTEGER, transaction_id TEXT, account_id TEXT, rule_ids TEXT[],
  triggered_at TIMESTAMP, status TEXT, enrichment JSONB)
  -- (enrichment->>'model_score')::numeric = fraud-risk score, 0 to 1, higher = riskier
  -- enrichment->>'narrative_summary' = plain-text explanation of why the alert fired
  -- enrichment->'structured_facts'->>'amount_zscore', ->>'velocity_24h_count', etc.

dispositions(disposition_id INTEGER, alert_id INTEGER, analyst_id TEXT,
  decision TEXT, notes TEXT, decided_at TIMESTAMP)
  -- decision is 'fraud' or 'not_fraud' -- the actual analyst-confirmed outcome

rules(rule_id TEXT, rule_name TEXT, description TEXT, created_at TIMESTAMP)

Join keys (important — these tables do NOT share an alert_id/account_id column
directly except where listed):
- alerts.transaction_id = analyst_transactions.transaction_id
- alerts.account_id = accounts.account_id
- dispositions.alert_id = alerts.alert_id
- analyst_transactions has NO alert_id column; join through alerts to reach it.

Rules for the SQL you write:
- Only SELECT statements. No INSERT/UPDATE/DELETE/DDL.
- Only use the tables/view listed above (use analyst_transactions, not transactions).
- Always include a LIMIT (200 max) unless the query is a pure aggregate (COUNT/AVG/SUM
  with no row-level SELECT columns).
- To rank by fraud risk: ORDER BY (enrichment->>'model_score')::numeric DESC
- CONFIRMED FRAUD is recorded in the dispositions table (decision = 'fraud'), NOT in
  analyst_transactions. To analyze confirmed-fraud transactions, join:
  dispositions d -> alerts a (a.alert_id = d.alert_id) -> analyst_transactions at
  (at.transaction_id = a.transaction_id), filtering WHERE d.decision = 'fraud'.
- NEVER divide by a column that can be zero (amount, balances often are 0). Wrap every
  divisor in NULLIF(divisor, 0), e.g. numerator / NULLIF(orig_balance_before, 0).
- Keep queries as SIMPLE as possible — one aggregation, minimal joins. Prefer several
  simple columns over one complex expression.

Worked examples (follow these patterns closely):

Q: How many confirmed fraud alerts are there?
SQL: SELECT COUNT(*) AS fraud_count FROM dispositions WHERE decision = 'fraud';

Q: What transaction types are most common among confirmed fraud?
SQL: SELECT at.transaction_type, COUNT(*) AS n
FROM dispositions d
JOIN alerts a ON a.alert_id = d.alert_id
JOIN analyst_transactions at ON at.transaction_id = a.transaction_id
WHERE d.decision = 'fraud'
GROUP BY at.transaction_type ORDER BY n DESC;

Q: What is the average transaction amount of confirmed fraud vs non-fraud?
SQL: SELECT d.decision, ROUND(AVG(at.amount), 2) AS avg_amount, COUNT(*) AS n
FROM dispositions d
JOIN alerts a ON a.alert_id = d.alert_id
JOIN analyst_transactions at ON at.transaction_id = a.transaction_id
GROUP BY d.decision;

Q: What is the common behavior of fraud? / How does fraud differ from legit activity?
SQL: SELECT d.decision,
  COUNT(*) AS n,
  ROUND(AVG(at.amount), 2) AS avg_amount,
  ROUND(AVG(CASE WHEN at.orig_balance_after = 0 AND at.orig_balance_before > 0 THEN 1.0 ELSE 0.0 END), 3) AS origin_emptied_rate,
  ROUND(AVG(CASE WHEN at.transaction_type IN ('TRANSFER','CASH_OUT') THEN 1.0 ELSE 0.0 END), 3) AS transfer_or_cashout_rate
FROM dispositions d
JOIN alerts a ON a.alert_id = d.alert_id
JOIN analyst_transactions at ON at.transaction_id = a.transaction_id
GROUP BY d.decision;
"""
