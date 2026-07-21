"""
Compact schema description fed to the SQL-generation prompt. Deliberately
scoped to exactly what the read-only role can see (see db/setup_readonly_role.sql):
analyst_transactions rather than the raw transactions table, which hides
is_fraud, and analyst_dispositions rather than dispositions, which hides the
seeded copies of that same label.

The worked examples below lean on model_score rather than on confirmed
outcomes. That is not a stylistic choice: analyst_dispositions contains only
decisions a human actually made, so on a freshly built database it is empty,
and examples written against decision='fraud' would teach the model to answer
every question with a query that returns nothing.
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
  triggered_at TIMESTAMP, status TEXT, model_score NUMERIC, enrichment JSONB)
  -- model_score = fraud-risk score, 0 to 1, higher = riskier. Prefer this column;
  --   (enrichment->>'model_score')::numeric holds the same value but is slower.
  -- status is 'open' (not yet reviewed) or 'closed' (an analyst dispositioned it)
  -- enrichment->>'narrative_summary' = plain-text explanation of why the alert fired
  -- enrichment->'structured_facts'->>'amount_zscore', ->>'velocity_24h_count', etc.
  -- enrichment->'reason_codes' = array of SHAP drivers behind the score

analyst_dispositions(disposition_id INTEGER, alert_id INTEGER, analyst_id TEXT,
  decision TEXT, notes TEXT, decided_at TIMESTAMP)
  -- decision is 'fraud' or 'not_fraud'
  -- IMPORTANT: this contains only reviews a human analyst actually completed. It
  --   is often EMPTY. "No alerts have been dispositioned yet" is a correct and
  --   expected answer -- do not treat zero rows as an error, and never imply a
  --   fraud total can be read from here when it returns nothing.

rules(rule_id TEXT, rule_name TEXT, description TEXT, created_at TIMESTAMP)

Join keys (important — these tables do NOT share an alert_id/account_id column
directly except where listed):
- alerts.transaction_id = analyst_transactions.transaction_id
- alerts.account_id = accounts.account_id
- analyst_dispositions.alert_id = alerts.alert_id
- analyst_transactions has NO alert_id column; join through alerts to reach it.

Rules for the SQL you write:
- Only SELECT statements. No INSERT/UPDATE/DELETE/DDL.
- Only use the tables/views listed above. In particular the tables `transactions`
  and `dispositions` are NOT available to you; use analyst_transactions and
  analyst_dispositions.
- Always include a LIMIT (200 max) unless the query is a pure aggregate (COUNT/AVG/SUM
  with no row-level SELECT columns).
- To rank by fraud risk: ORDER BY model_score DESC
- RISK is the model's score, which exists for every alert. CONFIRMED OUTCOME lives in
  analyst_dispositions and exists only where a human has reviewed. If a question asks
  about risk, likelihood, or what looks suspicious, use model_score. Only use
  analyst_dispositions when the question is specifically about analyst decisions,
  review progress, or what has been confirmed.
- NEVER divide by a column that can be zero (amount, balances often are 0). Wrap every
  divisor in NULLIF(divisor, 0), e.g. numerator / NULLIF(orig_balance_before, 0).
- Keep queries as SIMPLE as possible — one aggregation, minimal joins. Prefer several
  simple columns over one complex expression.

Worked examples (follow these patterns closely):

Q: How many high-risk alerts are there?
SQL: SELECT COUNT(*) AS high_risk FROM alerts WHERE model_score > 0.9;

Q: What transaction types appear most among high-risk alerts?
SQL: SELECT at.transaction_type, COUNT(*) AS n
FROM alerts a
JOIN analyst_transactions at ON at.transaction_id = a.transaction_id
WHERE a.model_score > 0.9
GROUP BY at.transaction_type ORDER BY n DESC;

Q: What do the riskiest alerts have in common?
SQL: SELECT at.transaction_type,
  COUNT(*) AS n,
  ROUND(AVG(at.amount), 2) AS avg_amount,
  ROUND(AVG(CASE WHEN at.orig_balance_after = 0 AND at.orig_balance_before > 0 THEN 1.0 ELSE 0.0 END), 3) AS origin_emptied_rate
FROM alerts a
JOIN analyst_transactions at ON at.transaction_id = a.transaction_id
WHERE a.model_score > 0.9
GROUP BY at.transaction_type ORDER BY n DESC;

Q: Show me the top 10 riskiest alerts.
SQL: SELECT a.alert_id, a.model_score, at.transaction_type, at.amount
FROM alerts a
JOIN analyst_transactions at ON at.transaction_id = a.transaction_id
ORDER BY a.model_score DESC LIMIT 10;

Q: Which rules fire most often?
SQL: SELECT unnest(rule_ids) AS rule_id, COUNT(*) AS n
FROM alerts GROUP BY 1 ORDER BY n DESC;

Q: How many alerts has the team reviewed so far?
SQL: SELECT decision, COUNT(*) AS n FROM analyst_dispositions GROUP BY decision;

Q: How many alerts have analysts confirmed as fraud?
SQL: SELECT COUNT(*) AS n
FROM analyst_dispositions ad
JOIN alerts a ON a.alert_id = ad.alert_id
WHERE ad.decision = 'fraud';

Note on aliases: if you alias a table you MUST use that alias on every column
reference, and you must alias every table you reference. Writing
"FROM alerts JOIN analyst_dispositions ON a.alert_id = ad.alert_id" is invalid
because neither table was given the alias it is then referred to by.
"""
