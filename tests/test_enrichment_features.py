"""
Feature-correctness tests.

The velocity tests exist because of a real bug: `compute_velocity_features`
sorted the frame by timestamp globally, then assigned the output of a
`groupby().rolling()` back positionally. groupby emits rows in
(account_id, ts) order, so every value landed on the wrong transaction —
silently, with plausible-looking numbers, costing ~0.15 ROC-AUC.

Two things make these tests able to catch that class of bug, and both are
deliberate:

  1. Assertions are keyed by transaction_id, never positional. A positional
     assertion would have agreed with the buggy implementation.
  2. The fixture interleaves two accounts in time, so global-ts order and
     (account_id, ts) order genuinely differ. If both accounts' rows were
     already contiguous, the misalignment would cancel out and the test would
     pass against broken code.
"""
import enrichment_features as ef
import numpy as np
import pandas as pd
import pytest

TS = pd.Timestamp("2024-01-01 00:00:00")


def _interleaved_frame() -> pd.DataFrame:
    """
    Two accounts alternating in time. Global-ts order is A,B,A,B,A while
    (account_id, ts) order is A,A,A,B,B — the exact condition that exposed
    the original misalignment.
    """
    rows = [
        ("a0", "ACC_A", TS + pd.Timedelta(hours=0), 100.0),
        ("b0", "ACC_B", TS + pd.Timedelta(hours=1), 999.0),
        ("a1", "ACC_A", TS + pd.Timedelta(hours=2), 200.0),
        ("b1", "ACC_B", TS + pd.Timedelta(hours=3), 888.0),
        ("a2", "ACC_A", TS + pd.Timedelta(hours=4), 300.0),
    ]
    return pd.DataFrame(rows, columns=["transaction_id", "account_id", "ts", "amount"])


def _by_txn(result: pd.DataFrame, column: str) -> dict:
    return dict(zip(result["transaction_id"], result[column], strict=True))


# --------------------------------------------------------------------------
# Velocity
# --------------------------------------------------------------------------
def test_velocity_values_land_on_the_correct_transactions():
    """The regression test for the misalignment bug. Keyed, not positional."""
    result = ef.compute_velocity_features(_interleaved_frame())

    sums = _by_txn(result, "velocity_24h_sum")
    counts = _by_txn(result, "velocity_24h_count")

    # ACC_A: trailing sums 100, 100+200, 100+200+300
    assert sums["a0"] == pytest.approx(100.0)
    assert sums["a1"] == pytest.approx(300.0)
    assert sums["a2"] == pytest.approx(600.0)

    # ACC_B: trailing sums 999, 999+888 — must NOT pick up any of ACC_A's amounts
    assert sums["b0"] == pytest.approx(999.0)
    assert sums["b1"] == pytest.approx(1887.0)

    assert counts["a0"] == 1 and counts["a1"] == 2 and counts["a2"] == 3
    assert counts["b0"] == 1 and counts["b1"] == 2


def test_velocity_is_invariant_to_input_row_order():
    """
    Same transactions, shuffled input, identical keyed output. An
    order-dependent implementation cannot satisfy this.
    """
    frame = _interleaved_frame()
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    baseline = _by_txn(ef.compute_velocity_features(frame), "velocity_24h_sum")
    reordered = _by_txn(ef.compute_velocity_features(shuffled), "velocity_24h_sum")

    assert baseline == reordered


def test_velocity_window_excludes_transactions_older_than_24h():
    frame = pd.DataFrame(
        [
            ("old", "ACC_A", TS, 500.0),
            ("new", "ACC_A", TS + pd.Timedelta(hours=25), 10.0),
        ],
        columns=["transaction_id", "account_id", "ts", "amount"],
    )
    result = ef.compute_velocity_features(frame)

    # 25h apart: the older transaction has rolled out of the window.
    assert _by_txn(result, "velocity_24h_sum")["new"] == pytest.approx(10.0)
    assert _by_txn(result, "velocity_24h_count")["new"] == 1


def test_velocity_handles_single_transaction_accounts():
    """
    Singletons take a shortcut path that skips the rolling call entirely, so
    it needs its own coverage — this is where the original bug hid, because
    count is trivially correct for singletons even when sum is not.
    """
    frame = pd.DataFrame(
        [
            ("s1", "ACC_ONE", TS, 42.0),
            ("s2", "ACC_TWO", TS + pd.Timedelta(hours=1), 77.0),
        ],
        columns=["transaction_id", "account_id", "ts", "amount"],
    )
    result = ef.compute_velocity_features(frame)

    assert _by_txn(result, "velocity_24h_count") == {"s1": 1.0, "s2": 1.0}
    assert _by_txn(result, "velocity_24h_sum") == {"s1": 42.0, "s2": 77.0}


# --------------------------------------------------------------------------
# Prior expanding statistics — vectorized vs. reference implementation
# --------------------------------------------------------------------------
def test_prior_expanding_stats_match_the_pandas_reference():
    """
    The cumsum implementation replaced `transform(lambda s: s.shift(1)
    .expanding()...)` for speed. This pins that the optimisation did not
    change semantics, including the NaN edges: mean undefined with 0 prior
    points, std (ddof=1) undefined with fewer than 2.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "account_id": ["A"] * 6 + ["B"] * 4 + ["C"],
            "amount": rng.uniform(1, 1000, 11),
        }
    )

    fast_mean, fast_std = ef._prior_expanding_mean_std(frame, "account_id", "amount")

    grouped = frame.groupby("account_id", sort=False)["amount"]
    ref_mean = grouped.transform(lambda s: s.shift(1).expanding().mean())
    ref_std = grouped.transform(lambda s: s.shift(1).expanding().std())

    pd.testing.assert_series_equal(fast_mean, ref_mean, check_names=False)
    pd.testing.assert_series_equal(fast_std, ref_std, check_names=False)


def test_amount_zscore_is_prior_only():
    """
    The z-score must not see the current transaction's own amount, or a large
    fraudulent amount would partly normalise itself away.
    """
    frame = pd.DataFrame(
        {
            "account_id": ["A", "A", "A", "A", "A"],
            "amount": [100.0, 120.0, 90.0, 110.0, 10_000.0],
        }
    )
    z = ef.compute_amount_zscore(frame)

    assert z.iloc[0] == 0.0, "no prior history -> no signal"
    assert z.iloc[4] > 100, "outlier must register against the prior-only baseline"


def test_amount_zscore_is_blind_to_outliers_after_constant_history():
    """
    KNOWN LIMITATION, pinned deliberately rather than fixed silently.

    With a perfectly constant prior history the prior std is 0, so the z-score
    is undefined and the code falls back to 0 — meaning the most suspicious
    possible pattern (a flat account suddenly moving 100x its usual amount)
    scores as *no signal*.

    This is not academic on PaySim: ~1M transactions span ~1M distinct origin
    accounts, so almost every account is a singleton, std is undefined, and
    amount_zscore is 0.0 for 100% of alerts in the loaded dataset. The feature
    is structurally dead here and the ranking model gets its signal from the
    balance-reconciliation features instead.

    Changing the fallback would alter feature semantics and require retraining,
    so the behaviour is documented and tested rather than quietly changed.
    """
    frame = pd.DataFrame(
        {
            "account_id": ["A", "A", "A", "A"],
            "amount": [100.0, 100.0, 100.0, 10_000.0],
        }
    )
    assert ef.compute_amount_zscore(frame).iloc[3] == 0.0


# --------------------------------------------------------------------------
# Prior alert statistics
# --------------------------------------------------------------------------
def test_prior_alert_stats_exclude_the_current_alert():
    """
    prior_fp_rate feeds the model, so leaking the current alert's own outcome
    into it would be target leakage.
    """
    alerts = pd.DataFrame(
        {
            "alert_id": [1, 2, 3, 4],
            "account_id": ["A", "A", "A", "B"],
            "triggered_at": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01"]
            ),
        }
    )
    dispositions = pd.DataFrame(
        {
            "alert_id": [1, 2, 3, 4],
            "decision": ["not_fraud", "not_fraud", "fraud", "fraud"],
        }
    )

    result = ef.compute_prior_alert_stats(alerts, dispositions).set_index("alert_id")

    # Alert 1 is account A's first: no prior alerts, no prior FP rate.
    assert result.loc[1, "prior_alert_count"] == 0
    assert result.loc[1, "prior_fp_rate"] == 0.0

    # Alert 2 sees exactly one prior alert, which was a false positive.
    assert result.loc[2, "prior_alert_count"] == 1
    assert result.loc[2, "prior_fp_rate"] == pytest.approx(1.0)

    # Alert 3 sees two priors, both false positives — its own 'fraud'
    # outcome must not be counted.
    assert result.loc[3, "prior_alert_count"] == 2
    assert result.loc[3, "prior_fp_rate"] == pytest.approx(1.0)

    # Account B's first alert must not inherit account A's history.
    assert result.loc[4, "prior_alert_count"] == 0


# --------------------------------------------------------------------------
# Narrative summary
# --------------------------------------------------------------------------
def test_narrative_summary_is_deterministic():
    """Template-based and auditable by design — not LLM-generated."""
    row = {"amount_zscore": 8.4, "velocity_24h_count": 6, "rule_ids": ["high_amount"]}
    assert ef.build_narrative_summary(row) == ef.build_narrative_summary(row)


def test_narrative_summary_handles_low_signal_alerts():
    assert "Low-signal" in ef.build_narrative_summary({})
