"""
Runs the full rule set against all loaded transactions and writes:
  - rules metadata (once)
  - alerts (one row per transaction that triggered >=1 rule, with all rule_ids attached)
  - dispositions (seeded from the dataset's ground-truth is_fraud label,
    standing in for "what the analyst determined" for Phase 1)

Usage:
    python rules/engine.py
    python rules/engine.py --skip-rules new_device,missing_identity_high_amount,region_mismatch
    (use --skip-rules for datasets without device/identity/region fields, e.g. PaySim)
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "db"))

import numpy as np
import pandas as pd
from db_utils import bulk_copy_insert, get_engine, pg_text_array
from rule_definitions import RULE_REGISTRY
from sqlalchemy import text

# FK constraints + secondary indexes on alerts/dispositions that make bulk COPY
# pathologically slow at multi-million-row scale: every inserted alert triggers
# two per-row FK probes against the ~6.3M-row transactions/accounts parents
# (disk-bound), plus unique/secondary index maintenance. We drop them, bulk-load,
# then recreate — Postgres validates each FK once as a single set-based join
# instead of millions of individual index lookups. (Primary keys are kept: they
# assign the SERIAL alert_id we need for disposition seeding, and appending
# sequential values to them is cheap.)
_DROP_BEFORE_LOAD = [
    "ALTER TABLE dispositions DROP CONSTRAINT IF EXISTS dispositions_alert_id_fkey",
    "ALTER TABLE alerts DROP CONSTRAINT IF EXISTS alerts_transaction_id_fkey",
    "ALTER TABLE alerts DROP CONSTRAINT IF EXISTS alerts_account_id_fkey",
    "ALTER TABLE alerts DROP CONSTRAINT IF EXISTS alerts_transaction_id_key",
    "DROP INDEX IF EXISTS idx_alerts_account",
    "DROP INDEX IF EXISTS idx_alerts_status",
    "DROP INDEX IF EXISTS idx_dispositions_alert",
]
_RECREATE_AFTER_LOAD = [
    "ALTER TABLE alerts ADD CONSTRAINT alerts_transaction_id_key UNIQUE (transaction_id)",
    "CREATE INDEX idx_alerts_account ON alerts (account_id)",
    "CREATE INDEX idx_alerts_status ON alerts (status)",
    "CREATE INDEX idx_dispositions_alert ON dispositions (alert_id)",
    # FKs added NOT VALID: every alert's transaction_id/account_id was drawn
    # directly from the transactions/accounts tables (and each disposition's
    # alert_id from the alerts we just inserted), so existing rows are correct by
    # construction. NOT VALID skips the full verification scan — which at 6M+ rows
    # would repeat exactly the disk-bound per-row probing we dropped the FKs to
    # avoid — while still enforcing referential integrity for any future writes.
    "ALTER TABLE alerts ADD CONSTRAINT alerts_transaction_id_fkey "
    "FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id) NOT VALID",
    "ALTER TABLE alerts ADD CONSTRAINT alerts_account_id_fkey "
    "FOREIGN KEY (account_id) REFERENCES accounts(account_id) NOT VALID",
    "ALTER TABLE dispositions ADD CONSTRAINT dispositions_alert_id_fkey "
    "FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) NOT VALID",
]


def _run_ddl(engine, statements, label):
    for i, stmt in enumerate(statements, 1):
        print(f"  [{i}/{len(statements)}] {stmt.split(chr(10))[0][:70]} ...")
        with engine.begin() as conn:
            conn.execute(text(stmt))
    print(f"{label} done.")


def load_data(engine):
    print("Loading transactions and accounts from Postgres ...")
    transactions = pd.read_sql("SELECT * FROM transactions ORDER BY account_id, ts", engine)
    accounts = pd.read_sql("SELECT * FROM accounts", engine)
    print(f"  {len(transactions):,} transactions, {len(accounts):,} accounts")
    return transactions, accounts


def register_rules(engine):
    """Write the rule catalogue once; a re-run of the engine leaves it alone."""
    if _rules_exist(engine):
        return
    rows = [
        {"rule_id": rid, "rule_name": rid.replace("_", " ").title(), "description": desc}
        for rid, (_, desc, _) in RULE_REGISTRY.items()
    ]
    pd.DataFrame(rows).to_sql("rules", engine, if_exists="append", index=False, method="multi")


def _rules_exist(engine) -> bool:
    existing = pd.read_sql("SELECT COUNT(*) AS n FROM rules", engine)
    return existing["n"].iat[0] > 0


def run_rules(transactions: pd.DataFrame, accounts: pd.DataFrame, skip_rules: set[str] = None) -> pd.DataFrame:
    print("Running rules ...")
    skip_rules = skip_rules or set()
    fired = pd.DataFrame(index=transactions.index)

    for rule_id, (fn, _desc, kwargs) in RULE_REGISTRY.items():
        if rule_id in skip_rules:
            print(f"  {rule_id} ... skipped (fields not available in this dataset)")
            fired[rule_id] = False
            continue
        print(f"  {rule_id} ...")
        if kwargs.get("needs_accounts"):
            result = fn(transactions, accounts)
        else:
            result = fn(transactions)
        fired[rule_id] = result

    transactions = transactions.copy()
    # Vectorized instead of fired.apply(..., axis=1): row-wise DataFrame.apply
    # builds a Series per row and is catastrophically slow at millions of rows;
    # raw numpy boolean indexing does the same job with no per-row Series overhead.
    fired_arr = fired.to_numpy()
    rule_cols = np.array(fired.columns)
    transactions["rule_ids"] = [rule_cols[row].tolist() for row in fired_arr]
    transactions["alert_flag"] = fired_arr.any(axis=1)
    return transactions


def write_alerts_and_dispositions(engine, transactions: pd.DataFrame):
    alerts_df = transactions[transactions["alert_flag"]].copy()
    print(f"Generated {len(alerts_df):,} alerts out of {len(transactions):,} transactions "
          f"({len(alerts_df) / len(transactions):.1%} alert rate)")

    alert_rows = alerts_df[["transaction_id", "account_id", "rule_ids", "ts"]].rename(
        columns={"ts": "triggered_at"}
    )
    alert_rows["status"] = "open"
    alert_rows["enrichment"] = None
    # rule_ids is a TEXT[] column — COPY needs it pre-serialized to Postgres
    # array-literal syntax rather than Python list repr.
    alert_rows["rule_ids"] = alert_rows["rule_ids"].apply(pg_text_array)

    # Drop FK constraints + secondary indexes so the bulk COPY isn't throttled by
    # per-row FK probes against the multi-million-row parent tables; recreated
    # (and FKs re-validated in one pass) after both tables are loaded.
    print("Dropping FK constraints + secondary indexes for fast bulk load ...")
    _run_ddl(engine, _DROP_BEFORE_LOAD, "  drop")

    print("Writing alerts (COPY) ...")
    # alert_id is a SERIAL PK, not in the column list below, so Postgres
    # assigns it from the sequence same as a normal INSERT would.
    # Capturing it via RETURNING would need per-row execution; simplest for
    # Phase 1 is bulk insert then re-select ids for disposition seeding.
    bulk_copy_insert(engine, alert_rows, "alerts",
                      ["transaction_id", "account_id", "rule_ids", "triggered_at", "status", "enrichment"])

    print("Seeding dispositions from ground-truth labels ...")
    alerts_in_db = pd.read_sql("SELECT alert_id, transaction_id FROM alerts", engine)
    merged = alerts_in_db.merge(
        transactions[["transaction_id", "is_fraud"]], on="transaction_id", how="left"
    )
    dispositions = pd.DataFrame({
        "alert_id": merged["alert_id"],
        "analyst_id": "seed_ground_truth",
        "decision": merged["is_fraud"].map({True: "fraud", False: "not_fraud"}),
        "notes": "Seeded from IEEE-CIS ground-truth label (Phase 1 bootstrap, not a live analyst decision)",
    })
    bulk_copy_insert(engine, dispositions, "dispositions", ["alert_id", "analyst_id", "decision", "notes"])

    print("Recreating indexes + FK constraints (single-pass validation) ...")
    _run_ddl(engine, _RECREATE_AFTER_LOAD, "  recreate")

    fraud_alert_rate = merged["is_fraud"].mean()
    print(f"Of generated alerts, {fraud_alert_rate:.1%} are actually fraud "
          f"=> false positive rate ~ {1 - fraud_alert_rate:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rules", type=str, default="",
                         help="Comma-separated rule_ids to skip, e.g. new_device,missing_identity_high_amount,region_mismatch")
    args = parser.parse_args()
    skip_rules = set(r.strip() for r in args.skip_rules.split(",") if r.strip())

    engine = get_engine()
    register_rules(engine)
    transactions, accounts = load_data(engine)
    transactions = run_rules(transactions, accounts, skip_rules=skip_rules)
    write_alerts_and_dispositions(engine, transactions)
    print("Done.")


if __name__ == "__main__":
    main()
