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

GRANT CONNECT ON DATABASE fraud_intel TO fraud_intel_readonly;
GRANT USAGE ON SCHEMA public TO fraud_intel_readonly;
GRANT SELECT ON accounts, alerts, dispositions, rules, analyst_transactions TO fraud_intel_readonly;
REVOKE ALL ON transactions FROM fraud_intel_readonly;
