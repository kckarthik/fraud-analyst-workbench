-- Fraud Intelligence Platform — Phase 1 schema
-- Run this once against a fresh database before load_data.py

DROP TABLE IF EXISTS dispositions CASCADE;
DROP TABLE IF EXISTS alerts CASCADE;
DROP TABLE IF EXISTS rules CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS accounts CASCADE;

CREATE TABLE accounts (
    account_id       TEXT PRIMARY KEY,      -- proxy: derived from IEEE-CIS card1
    first_seen_at    TIMESTAMP,             -- proxy for account open date (first transaction seen)
    account_type     TEXT,                  -- proxy: card6 (debit/credit)
    card_network     TEXT,                  -- proxy: card4 (visa/mastercard/...)
    region_code      TEXT                   -- proxy: most common addr1 for this account
);

CREATE TABLE transactions (
    transaction_id       TEXT PRIMARY KEY,      -- IEEE-CIS TransactionID
    account_id            TEXT REFERENCES accounts(account_id),
    ts                     TIMESTAMP NOT NULL,    -- derived from TransactionDT offset
    amount                 NUMERIC NOT NULL,
    currency               TEXT DEFAULT 'USD',
    transaction_type       TEXT,                  -- proxy: ProductCD
    counterparty_id        TEXT,                  -- proxy: P_emaildomain
    counterparty_region    TEXT,                  -- proxy: addr2
    device_id              TEXT,                  -- proxy: DeviceInfo (identity table, nullable)
    ip_proxy                TEXT,                  -- proxy: hash of id_19/id_20 (identity table, nullable)
    has_identity_data       BOOLEAN DEFAULT FALSE,
    orig_balance_before     NUMERIC,               -- PaySim oldbalanceOrg (origin acct balance pre-txn; NULL for datasets without it)
    orig_balance_after      NUMERIC,               -- PaySim newbalanceOrig
    dest_balance_before     NUMERIC,               -- PaySim oldbalanceDest
    dest_balance_after      NUMERIC,               -- PaySim newbalanceDest
    is_fraud                BOOLEAN NOT NULL       -- ground truth label, hidden from analyst-facing views
);

CREATE INDEX idx_transactions_account_ts ON transactions (account_id, ts);
CREATE INDEX idx_transactions_device ON transactions (device_id);
CREATE INDEX idx_transactions_counterparty ON transactions (counterparty_id);

CREATE TABLE rules (
    rule_id          TEXT PRIMARY KEY,
    rule_name         TEXT NOT NULL,
    description        TEXT,
    created_at         TIMESTAMP DEFAULT now()
);

CREATE TABLE alerts (
    alert_id          SERIAL PRIMARY KEY,
    transaction_id      TEXT REFERENCES transactions(transaction_id),
    account_id           TEXT REFERENCES accounts(account_id),
    rule_ids              TEXT[] NOT NULL,        -- all rules that fired for this transaction
    triggered_at           TIMESTAMP NOT NULL,
    status                  TEXT DEFAULT 'open',    -- open / closed
    enrichment              JSONB,                  -- structured_facts + narrative_summary + model_score + reason_codes (Phase 2)
    model_score              NUMERIC,                -- materialized copy of enrichment->>'model_score' (Phase 2/3):
                                                       -- a functional index on the JSONB expression was tried first and
                                                       -- the planner refused to use it once a JOIN + parallel workers were
                                                       -- involved (fell back to a multi-hundred-MB disk sort) — a real
                                                       -- column with a plain index is what makes the ranked-queue query fast.
                                                       -- Kept in sync by model/explain.py after each JSONB write-back.
    UNIQUE (transaction_id)                          -- one alert per transaction for v1 (multiple rules collapse into one alert)
);

CREATE INDEX idx_alerts_account ON alerts (account_id);
CREATE INDEX idx_alerts_status ON alerts (status);
CREATE INDEX idx_alerts_model_score ON alerts (model_score DESC NULLS LAST);

CREATE TABLE dispositions (
    disposition_id     SERIAL PRIMARY KEY,
    alert_id             INTEGER REFERENCES alerts(alert_id),
    analyst_id             TEXT DEFAULT 'seed_ground_truth',  -- Phase 1: backfilled from dataset label
    decision                TEXT NOT NULL,                     -- 'fraud' / 'not_fraud'
    notes                    TEXT,
    decided_at               TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_dispositions_alert ON dispositions (alert_id);

-- Phase 3: read-only role + view for the NL2SQL agent (see db/setup_readonly_role.sql
-- for the idempotent standalone version that doesn't require a full schema reset).
--
-- The role password is supplied at run time, never hardcoded. Run this file with:
--   docker exec -i fraud-intel-db psql -U postgres -d fraud_intel \
--     -v readonly_password="$DB_READONLY_PASSWORD" < db/schema.sql
\if :{?readonly_password}
\else
\echo ''
\echo 'ERROR: psql variable "readonly_password" is not set.'
\echo 'Re-run with: -v readonly_password="$DB_READONLY_PASSWORD"'
\echo ''
\quit
\endif

-- psql does not interpolate :'vars' inside dollar-quoted blocks, so \gset + \if
-- replaces the DO-block idiom here. Kept in sync with setup_readonly_role.sql.
SELECT NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'fraud_intel_readonly') AS create_ro_role \gset

\if :create_ro_role
CREATE ROLE fraud_intel_readonly LOGIN;
\endif

ALTER ROLE fraud_intel_readonly WITH LOGIN PASSWORD :'readonly_password';

CREATE OR REPLACE VIEW analyst_transactions AS
SELECT transaction_id, account_id, ts, amount, currency, transaction_type,
       counterparty_id, counterparty_region, device_id, ip_proxy, has_identity_data,
       orig_balance_before, orig_balance_after, dest_balance_before, dest_balance_after
FROM transactions;

GRANT CONNECT ON DATABASE fraud_intel TO fraud_intel_readonly;
GRANT USAGE ON SCHEMA public TO fraud_intel_readonly;
GRANT SELECT ON accounts, alerts, dispositions, rules, analyst_transactions TO fraud_intel_readonly;
REVOKE ALL ON transactions FROM fraud_intel_readonly;
