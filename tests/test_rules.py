"""
Rule correctness tests, focused on point-in-time validity.

A fraud rule is evaluated on a transaction as it arrives, so it may only depend
on that transaction and ones already seen. A rule that reads the account's whole
day looks perfectly healthy in a batch backtest — every row gets the day's full
picture — and then cannot be computed at all in production, because the future
half of the day has not happened.

rule_multi_product_same_day was written that way. It used
`transform("nunique")` across the entire (account, day) group, so the first
transaction of the day already knew which other product types the account would
use later. Nothing failed; the batch pipeline simply produced a feature the
serving path could never reproduce. It surfaced only when the same transactions
were replayed through POST /api/score and the fired rules were diffed.
"""
import pandas as pd
import rule_definitions as rd

DAY = pd.Timestamp("2024-03-01")


def _frame(rows) -> pd.DataFrame:
    """rows: (transaction_id, account_id, ts, transaction_type, amount)."""
    df = pd.DataFrame(rows, columns=["transaction_id", "account_id", "ts", "transaction_type", "amount"])
    # Every rule assumes (account_id, ts) ordering, as engine.py guarantees.
    return df.sort_values(["account_id", "ts"]).reset_index(drop=True)


def _fired_by_txn(df: pd.DataFrame, result) -> dict:
    return dict(zip(df["transaction_id"], [bool(v) for v in result], strict=True))


class TestMultiProductSameDay:
    def test_first_transaction_cannot_see_a_later_product(self):
        """
        The regression. PAYMENT at 10:00, TRANSFER at 15:00, same account, same
        day. At 10:00 only one product type has been used, so the rule must not
        fire — the old implementation fired on both rows because nunique spanned
        the whole day.
        """
        df = _frame([
            ("t1", "ACC", DAY + pd.Timedelta(hours=10), "PAYMENT", 100.0),
            ("t2", "ACC", DAY + pd.Timedelta(hours=15), "TRANSFER", 200.0),
        ])

        fired = _fired_by_txn(df, rd.rule_multi_product_same_day(df))

        assert fired["t1"] is False, "10:00 transaction must not see the 15:00 product type"
        assert fired["t2"] is True

    def test_same_type_twice_never_fires(self):
        df = _frame([
            ("t1", "ACC", DAY + pd.Timedelta(hours=9), "PAYMENT", 10.0),
            ("t2", "ACC", DAY + pd.Timedelta(hours=11), "PAYMENT", 20.0),
            ("t3", "ACC", DAY + pd.Timedelta(hours=13), "PAYMENT", 30.0),
        ])

        assert not any(rd.rule_multi_product_same_day(df))

    def test_stays_fired_once_a_second_type_appears(self):
        """Third transaction reverts to the original type, but two distinct types have still been seen."""
        df = _frame([
            ("t1", "ACC", DAY + pd.Timedelta(hours=9), "PAYMENT", 10.0),
            ("t2", "ACC", DAY + pd.Timedelta(hours=11), "TRANSFER", 20.0),
            ("t3", "ACC", DAY + pd.Timedelta(hours=13), "PAYMENT", 30.0),
        ])

        fired = _fired_by_txn(df, rd.rule_multi_product_same_day(df))

        assert fired == {"t1": False, "t2": True, "t3": True}

    def test_does_not_carry_across_calendar_days(self):
        df = _frame([
            ("t1", "ACC", DAY + pd.Timedelta(hours=23), "PAYMENT", 10.0),
            ("t2", "ACC", DAY + pd.Timedelta(hours=25), "TRANSFER", 20.0),  # next day
        ])

        fired = _fired_by_txn(df, rd.rule_multi_product_same_day(df))

        assert fired == {"t1": False, "t2": False}

    def test_accounts_do_not_leak_into_each_other(self):
        """Two accounts interleaved in time, each using a single product type."""
        df = _frame([
            ("a1", "ACC_A", DAY + pd.Timedelta(hours=9), "PAYMENT", 10.0),
            ("b1", "ACC_B", DAY + pd.Timedelta(hours=10), "TRANSFER", 20.0),
            ("a2", "ACC_A", DAY + pd.Timedelta(hours=11), "PAYMENT", 30.0),
            ("b2", "ACC_B", DAY + pd.Timedelta(hours=12), "TRANSFER", 40.0),
        ])

        assert not any(rd.rule_multi_product_same_day(df))

    def test_single_transaction_account_never_fires(self):
        df = _frame([("t1", "ACC", DAY + pd.Timedelta(hours=9), "PAYMENT", 10.0)])
        assert not any(rd.rule_multi_product_same_day(df))
