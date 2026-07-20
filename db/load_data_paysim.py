"""
Loads the PaySim synthetic mobile-money dataset into the same fraud_intel
schema used for IEEE-CIS. PaySim has no device/identity/region fields, so
those columns are left NULL here (device/region-based rules are skipped
when running the rules engine against this dataset — see engine.py --skip-rules).

Expected file in ../data/: the PaySim CSV, commonly named
PS_20174392719_1491204439457_log.csv (Kaggle's default export name).
Adjust FILENAME below if yours is named differently.

Usage:
    python db/load_data_paysim.py                # full dataset (~6.3M rows)
    python db/load_data_paysim.py --sample 100000 # fast iteration
"""
import argparse
import glob
from datetime import datetime, timedelta

import pandas as pd

from db_utils import get_engine, bulk_copy_insert

DATA_DIR = "../data"
REFERENCE_START = datetime(2019, 1, 1)  # arbitrary anchor; PaySim's "step" = hours elapsed


def find_paysim_file() -> str:
    candidates = glob.glob(f"{DATA_DIR}/PS_*.csv") + glob.glob(f"{DATA_DIR}/paysim*.csv")
    if not candidates:
        raise FileNotFoundError(
            f"No PaySim CSV found in {DATA_DIR}/. "
            "Expected something like PS_20174392719_1491204439457_log.csv — "
            "rename your file to start with 'PS_' or 'paysim' if needed."
        )
    return candidates[0]


def load_raw(sample: int | None) -> pd.DataFrame:
    path = find_paysim_file()
    print(f"Reading {path} ...")
    df = pd.read_csv(path, nrows=sample)
    print(f"  {len(df):,} transactions loaded")
    return df


def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    print("Building transactions table ...")
    out = pd.DataFrame()
    out["transaction_id"] = ["paysim_" + str(i) for i in df.index]
    out["account_id"] = df["nameOrig"]
    out["ts"] = df["step"].apply(lambda h: REFERENCE_START + timedelta(hours=int(h)))
    out["amount"] = df["amount"]
    out["currency"] = "USD"
    out["transaction_type"] = df["type"]  # CASH_IN / CASH_OUT / DEBIT / PAYMENT / TRANSFER
    out["counterparty_id"] = df["nameDest"]
    out["counterparty_region"] = None       # not available in PaySim
    out["device_id"] = None                 # not available in PaySim
    out["ip_proxy"] = None                  # not available in PaySim
    out["has_identity_data"] = False        # PaySim never has identity/device data
    # Balance columns — PaySim's strongest fraud signal (fraud typically drains the
    # origin account to zero and the arithmetic often doesn't reconcile).
    out["orig_balance_before"] = df["oldbalanceOrg"]
    out["orig_balance_after"] = df["newbalanceOrig"]
    out["dest_balance_before"] = df["oldbalanceDest"]
    out["dest_balance_after"] = df["newbalanceDest"]
    out["is_fraud"] = df["isFraud"].astype(bool)
    return out


def build_accounts(df: pd.DataFrame) -> pd.DataFrame:
    print("Building accounts table ...")
    grp = df.groupby("nameOrig")
    accounts = grp.agg(
        first_seen_at=("step", lambda s: REFERENCE_START + timedelta(hours=int(s.min()))),
    ).reset_index().rename(columns={"nameOrig": "account_id"})
    accounts["account_type"] = "mobile_money"
    accounts["card_network"] = None
    accounts["region_code"] = None
    return accounts


def write_to_db(accounts: pd.DataFrame, transactions: pd.DataFrame):
    engine = get_engine()

    print(f"Writing {len(accounts):,} accounts (COPY) ...")
    bulk_copy_insert(engine, accounts, "accounts",
                      ["account_id", "first_seen_at", "account_type", "card_network", "region_code"])

    print(f"Writing {len(transactions):,} transactions (COPY) ...")
    bulk_copy_insert(engine, transactions, "transactions",
                      ["transaction_id", "account_id", "ts", "amount", "currency", "transaction_type",
                       "counterparty_id", "counterparty_region", "device_id", "ip_proxy",
                       "has_identity_data", "orig_balance_before", "orig_balance_after",
                       "dest_balance_before", "dest_balance_after", "is_fraud"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                         help="Limit to first N rows for fast local iteration")
    args = parser.parse_args()

    df = load_raw(args.sample)

    # Accounts must exist before transactions (FK constraint) — but PaySim's
    # nameDest values are never inserted as accounts, only nameOrig senders are.
    # A transaction's counterparty_id may reference an account not in our
    # accounts table, which is fine: counterparty_id is not a foreign key.
    accounts = build_accounts(df)
    transactions = build_transactions(df)

    write_to_db(accounts, transactions)

    print("Done.")
    print(f"  Accounts: {len(accounts):,}")
    print(f"  Transactions: {len(transactions):,}")
    print(f"  Fraud rate: {transactions['is_fraud'].mean():.3%}")


if __name__ == "__main__":
    main()
