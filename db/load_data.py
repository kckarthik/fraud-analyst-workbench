"""
Loads IEEE-CIS Fraud Detection data into the fraud_intel Postgres schema.

Expects these files in ../data/ (download from Kaggle first):
    train_transaction.csv
    train_identity.csv

Usage:
    python db/load_data.py                # full dataset
    python db/load_data.py --sample 50000  # first N transactions, for fast iteration
"""
import argparse
import hashlib
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from db_utils import get_engine

DATA_DIR = "../data"

# IEEE-CIS's TransactionDT is a seconds-offset from an unspecified reference point,
# not a real calendar date. We anchor it to an arbitrary reference date so we get
# usable, orderable timestamps. Absolute dates are not meaningful here — only
# relative ordering and deltas between transactions are.
REFERENCE_START = datetime(2019, 1, 1)


def load_raw(sample: int | None):
    print("Reading train_transaction.csv ...")
    txn = pd.read_csv(f"{DATA_DIR}/train_transaction.csv", nrows=sample)
    print(f"  {len(txn):,} transactions loaded")

    print("Reading train_identity.csv ...")
    identity = pd.read_csv(f"{DATA_DIR}/train_identity.csv")
    print(f"  {len(identity):,} identity rows loaded")

    print("Merging on TransactionID (left join — most transactions have no identity row) ...")
    df = txn.merge(identity, on="TransactionID", how="left")
    df["has_identity_data"] = df["DeviceInfo"].notna() | df["id_20"].notna()
    return df


def hash_ip_proxy(row):
    """
    IEEE-CIS doesn't expose real IP addresses — id_19/id_20 are anonymized
    numeric features loosely related to network/IP info. We combine them into
    a stable pseudo-IP identifier so device/IP-sharing rules have something
    consistent to key off. This is a proxy, not a real IP — documented as such.
    """
    if pd.isna(row.get("id_19")) and pd.isna(row.get("id_20")):
        return None
    raw = f"{row.get('id_19')}_{row.get('id_20')}"
    return "ipproxy_" + hashlib.sha1(raw.encode()).hexdigest()[:12]


def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    print("Building transactions table ...")
    out = pd.DataFrame()
    out["transaction_id"] = df["TransactionID"].astype(str)
    out["account_id"] = df["card1"].astype(str)  # proxy: card fingerprint as account
    out["ts"] = df["TransactionDT"].apply(lambda s: REFERENCE_START + timedelta(seconds=int(s)))
    out["amount"] = df["TransactionAmt"]
    out["currency"] = "USD"
    out["transaction_type"] = df["ProductCD"]
    out["counterparty_id"] = df["P_emaildomain"].fillna("unknown_counterparty")
    out["counterparty_region"] = df["addr2"].astype("Int64").astype(str)
    out["device_id"] = df["DeviceInfo"]
    out["ip_proxy"] = df.apply(hash_ip_proxy, axis=1)
    out["has_identity_data"] = df["has_identity_data"]
    out["is_fraud"] = df["isFraud"].astype(bool)
    return out


def build_accounts(df: pd.DataFrame) -> pd.DataFrame:
    print("Building accounts table ...")
    grp = df.groupby(df["card1"].astype(str))
    accounts = grp.agg(
        first_seen_at=("TransactionDT", lambda s: REFERENCE_START + timedelta(seconds=int(s.min()))),
        account_type=("card6", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        card_network=("card4", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        region_code=("addr1", lambda s: str(int(s.mode().iat[0])) if not s.mode().empty else None),
    ).reset_index().rename(columns={"card1": "account_id"})
    return accounts


def write_to_db(accounts: pd.DataFrame, transactions: pd.DataFrame):
    engine = get_engine()
    print(f"Writing {len(accounts):,} accounts ...")
    accounts.to_sql("accounts", engine, if_exists="append", index=False, chunksize=5000, method="multi")

    print(f"Writing {len(transactions):,} transactions (chunked) ...")
    chunk_size = 5000
    for start in tqdm(range(0, len(transactions), chunk_size)):
        chunk = transactions.iloc[start:start + chunk_size]
        chunk.to_sql("transactions", engine, if_exists="append", index=False, method="multi")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Limit to first N transactions for fast local iteration")
    args = parser.parse_args()

    df = load_raw(args.sample)
    accounts = build_accounts(df)
    transactions = build_transactions(df)
    write_to_db(accounts, transactions)

    print("Done.")
    print(f"  Accounts: {len(accounts):,}")
    print(f"  Transactions: {len(transactions):,}")
    print(f"  Fraud rate: {transactions['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()
