-- One-time, idempotent setup for the NL2SQL agent's DB access.
-- Safe to re-run: does not touch existing data.
--
-- The role password is supplied at run time, never hardcoded here. Run with:
--
--   PowerShell:
--     docker exec -i fraud-intel-db psql -U postgres -d fraud_intel `
--       -v readonly_password="$env:DB_READONLY_PASSWORD" < db\setup_readonly_role.sql
--
--   bash:
--     docker exec -i fraud-intel-db psql -U postgres -d fraud_intel \
--       -v readonly_password="$DB_READONLY_PASSWORD" < db/setup_readonly_role.sql

\set ON_ERROR_STOP on

\if :{?readonly_password}
\else
\echo ''
\echo 'ERROR: psql variable "readonly_password" is not set.'
\echo 'Re-run with: -v readonly_password="$DB_READONLY_PASSWORD" (see header of this file).'
\echo ''
\quit
\endif

-- NOTE: psql does not interpolate :'vars' inside dollar-quoted blocks ($$...$$),
-- so the usual DO-block idiom cannot be used to set the password. \gset + \if
-- keeps this idempotent while letting the password stay a psql variable.
SELECT NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'fraud_intel_readonly') AS create_ro_role \gset

\if :create_ro_role
CREATE ROLE fraud_intel_readonly LOGIN;
\endif

ALTER ROLE fraud_intel_readonly WITH LOGIN PASSWORD :'readonly_password';

-- Analyst-facing view: excludes is_fraud, the ground-truth label that's meant
-- to be hidden from analyst-facing views per the schema's own design intent.
CREATE OR REPLACE VIEW analyst_transactions AS
SELECT transaction_id, account_id, ts, amount, currency, transaction_type,
       counterparty_id, counterparty_region, device_id, ip_proxy, has_identity_data,
       orig_balance_before, orig_balance_after, dest_balance_before, dest_balance_after
FROM transactions;

-- Analyst-facing dispositions: excludes the rows rules/engine.py backfills from
-- the dataset's is_fraud label under analyst_id 'seed_ground_truth'.
--
-- Without this the label revocation below is decorative. Those seeded rows are
-- a verbatim copy of transactions.is_fraud for every alert, so an agent denied
-- the raw column could still recover the answer for any alert with
--   SELECT decision FROM dispositions WHERE alert_id = ...
-- and in practice did: asked "how many confirmed fraud alerts are there", it
-- wrote COUNT(*) FROM dispositions WHERE decision='fraud' and read back 523.
-- IS DISTINCT FROM, not <>, so a NULL analyst_id counts as a real review rather
-- than being silently swallowed.
CREATE OR REPLACE VIEW analyst_dispositions AS
SELECT disposition_id, alert_id, analyst_id, decision, notes, decided_at
FROM dispositions
WHERE analyst_id IS DISTINCT FROM 'seed_ground_truth';

GRANT CONNECT ON DATABASE fraud_intel TO fraud_intel_readonly;
GRANT USAGE ON SCHEMA public TO fraud_intel_readonly;
GRANT SELECT ON accounts, alerts, rules, analyst_transactions, analyst_dispositions
  TO fraud_intel_readonly;
REVOKE ALL ON transactions FROM fraud_intel_readonly;
-- Explicit, and not redundant: earlier revisions granted SELECT on dispositions,
-- so re-running this file against an existing database must take it back.
REVOKE ALL ON dispositions FROM fraud_intel_readonly;
