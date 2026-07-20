"""
Builds the enrichment JSON object for every alert and writes it back to
alerts.enrichment (JSONB column).

Usage:
    python enrichment/pipeline.py
"""
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "db"))

import pandas as pd
from tqdm import tqdm
from db_utils import get_engine, bulk_jsonb_update
from features import compute_velocity_features, compute_amount_zscore, compute_prior_alert_stats, build_narrative_summary


def load_data(engine):
    print("Loading alerts, transactions, dispositions ...")
    alerts = pd.read_sql("SELECT alert_id, transaction_id, account_id, rule_ids, triggered_at FROM alerts", engine)
    transactions = pd.read_sql("SELECT * FROM transactions ORDER BY account_id, ts", engine)
    dispositions = pd.read_sql("SELECT * FROM dispositions", engine)
    return alerts, transactions, dispositions


def build_enrichment(alerts: pd.DataFrame, transactions: pd.DataFrame, dispositions: pd.DataFrame) -> pd.DataFrame:
    print("Computing velocity features ...")
    txn_enriched = compute_velocity_features(transactions)
    txn_enriched["amount_zscore"] = compute_amount_zscore(txn_enriched)

    alerts_with_txn = alerts.merge(
        txn_enriched[["transaction_id", "velocity_24h_count", "velocity_24h_sum", "amount_zscore", "amount"]],
        on="transaction_id", how="left"
    )

    print("Computing prior alert / false-positive history ...")
    prior_stats = compute_prior_alert_stats(alerts, dispositions)
    alerts_with_txn = alerts_with_txn.merge(prior_stats, on="alert_id", how="left")

    print("Assembling structured_facts + narrative summaries ...")
    records = []
    # itertuples instead of iterrows: iterrows builds a full pandas Series
    # (with dtype coercion) per row, which is extremely slow at millions of
    # rows; itertuples yields lightweight namedtuples with no such overhead.
    for row in tqdm(alerts_with_txn.itertuples(index=False), total=len(alerts_with_txn)):
        facts = {
            "amount": float(row.amount) if pd.notna(row.amount) else None,
            "amount_zscore": round(float(row.amount_zscore), 2) if pd.notna(row.amount_zscore) else None,
            "velocity_24h_count": int(row.velocity_24h_count) if pd.notna(row.velocity_24h_count) else None,
            "velocity_24h_sum": round(float(row.velocity_24h_sum), 2) if pd.notna(row.velocity_24h_sum) else None,
            "prior_alert_count": int(row.prior_alert_count) if pd.notna(row.prior_alert_count) else 0,
            "prior_fp_rate": round(float(row.prior_fp_rate), 3) if pd.notna(row.prior_fp_rate) else 0.0,
            "rule_ids": row.rule_ids,
        }
        narrative = build_narrative_summary(facts)
        enrichment = {"structured_facts": facts, "narrative_summary": narrative}
        records.append({"alert_id": row.alert_id, "enrichment": json.dumps(enrichment)})

    return pd.DataFrame(records)


def write_enrichment(engine, enrichment_df: pd.DataFrame):
    """
    Bulk write via COPY-into-temp-table + a single UPDATE ... FROM join
    (see db_utils.bulk_jsonb_update) — ~an order of magnitude faster than
    per-chunk VALUES updates once this runs against the full dataset. This is
    the first enrichment write for each alert, so overwrite mode is correct.
    """
    print(f"Writing enrichment for {len(enrichment_df):,} alerts (bulk) ...")
    records = zip(enrichment_df["alert_id"].astype(int), enrichment_df["enrichment"])
    bulk_jsonb_update(engine, records, "alerts", "alert_id", "enrichment", mode="overwrite")


def main():
    engine = get_engine()
    alerts, transactions, dispositions = load_data(engine)
    enrichment_df = build_enrichment(alerts, transactions, dispositions)
    write_enrichment(engine, enrichment_df)
    print("Done.")


if __name__ == "__main__":
    main()
