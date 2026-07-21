"""
Queue-depth evaluation tests.

These cover the reporting layer rather than the model, because the reporting is
what the README's headline claims are read off — and the failure mode here is
not a crash, it's a plausible-looking number that overstates the result. The
cases below pin the three ways that can happen:

  1. An off-by-one at the cutoff (does "top K" include the Kth alert?).
  2. Ties at the top of the queue resolving differently between runs, which on
     this data affects thousands of alerts that all score >= 0.999.
  3. Recall quoted to three significant figures off a few dozen positives.
"""
import model_evaluate as ev
import numpy as np
import pandas as pd
import pytest


def _labels(*values) -> pd.Series:
    return pd.Series(list(values), dtype=int)


class TestQueueDepthAtK:
    def test_counts_fraud_within_top_k_inclusive(self):
        # Scores descending: the two fraud alerts sit at ranks 1 and 3.
        y = _labels(1, 0, 1, 0, 0)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        report = ev.queue_depth_report_at_k(y, scores, top_k=(1, 2, 3))
        caught = dict(zip(report["top_k"], report["fraud_caught"], strict=True))

        # top_k=3 must include the rank-3 alert, not stop just short of it.
        assert caught == {1: 1, 2: 1, 3: 2}

    def test_recall_and_precision_are_consistent_at_each_k(self):
        y = _labels(1, 0, 1, 0, 0)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        row = ev.queue_depth_report_at_k(y, scores, top_k=(3,)).iloc[0]

        assert row["fraud_caught"] == 2
        assert row["fraud_missed"] == 0
        assert row["total_fraud_in_test"] == 2
        assert row["precision_at_k"] == pytest.approx(2 / 3, abs=1e-4)
        assert row["recall_at_k"] == pytest.approx(1.0)

    def test_cutoffs_deeper_than_the_queue_are_dropped(self):
        """
        Guards against emitting a duplicate full-queue row for every K past the
        end — which would render as a run of identical 100% rows in the README
        and read as though deeper review kept helping.
        """
        y = _labels(1, 0, 0)
        scores = np.array([0.9, 0.5, 0.1])

        report = ev.queue_depth_report_at_k(y, scores, top_k=(2, 3, 100, 5000))

        assert list(report["top_k"]) == [2, 3]

    def test_ties_are_broken_stably_not_by_sort_implementation(self):
        """
        Every alert scores identically, so the report is entirely determined by
        tie-breaking. A stable sort keeps incoming order, making the number
        reproducible; an unstable one would let it vary between numpy versions.
        """
        y = _labels(0, 0, 1, 0)
        scores = np.full(4, 0.999)

        first = ev.queue_depth_report_at_k(y, scores, top_k=(2,))
        again = ev.queue_depth_report_at_k(y, scores, top_k=(2,))

        # The fraud alert is at incoming index 2, so it is NOT in a stable top-2.
        assert first.iloc[0]["fraud_caught"] == 0
        pd.testing.assert_frame_equal(first, again)

    def test_random_expectation_is_reported_alongside(self):
        # 2 fraud in 10 alerts => base rate 0.2 => top-5 random expectation 1.0
        y = _labels(1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
        scores = np.linspace(1.0, 0.1, 10)

        row = ev.queue_depth_report_at_k(y, scores, top_k=(5,)).iloc[0]

        assert row["random_expected_caught"] == pytest.approx(1.0)
        assert row["fraud_caught"] == 2

    def test_no_fraud_in_split_does_not_divide_by_zero(self):
        y = _labels(0, 0, 0)
        scores = np.array([0.9, 0.5, 0.1])

        row = ev.queue_depth_report_at_k(y, scores, top_k=(2,)).iloc[0]

        assert row["fraud_caught"] == 0
        assert row["recall_at_k"] is None


class TestWilsonInterval:
    def test_brackets_the_point_estimate(self):
        low, high = ev.wilson_interval(68, 69)
        assert low < 68 / 69 < high

    def test_small_sample_interval_is_wide_enough_to_matter(self):
        """
        The reason this exists: 68/69 reads as "98.6%", which implies a precision
        the sample cannot support. The interval has to be wide enough that
        quoting three significant figures is visibly wrong.
        """
        low, high = ev.wilson_interval(68, 69)
        assert low < 0.95
        assert high < 1.0

    def test_tightens_as_the_sample_grows(self):
        narrow_lo, narrow_hi = ev.wilson_interval(9860, 10000)
        wide_lo, wide_hi = ev.wilson_interval(68, 69)
        assert (narrow_hi - narrow_lo) < (wide_hi - wide_lo)

    def test_stays_within_zero_and_one_at_the_boundaries(self):
        assert ev.wilson_interval(10, 10)[1] <= 1.0
        assert ev.wilson_interval(0, 10)[0] >= 0.0

    def test_empty_sample_is_nan_not_an_exception(self):
        low, high = ev.wilson_interval(0, 0)
        assert np.isnan(low) and np.isnan(high)


class TestPercentageDepthReport:
    def test_unchanged_behaviour_at_fractional_depths(self):
        """The percentage report is still published, so its arithmetic stays pinned."""
        y = _labels(*([1] * 5 + [0] * 95))
        scores = np.linspace(1.0, 0.0, 100)

        row = ev.queue_depth_report(y, scores, depths=(0.05,)).iloc[0]

        assert row["alerts_reviewed"] == 5
        assert row["fraud_caught"] == 5
        assert row["recall_at_depth"] == pytest.approx(1.0)

    def test_random_baseline_recall_equals_the_depth(self):
        y = _labels(*([1] * 10 + [0] * 90))
        report = ev.random_baseline_report(y, depths=(0.1, 0.5))
        assert list(report["expected_recall_at_depth"]) == [0.1, 0.5]


class TestAmountSortedBaseline:
    def test_ranks_by_amount_not_by_model_score(self):
        """
        A team without a model works the biggest transactions first. Here the
        fraud is on the *smallest* amount, so an amount-sorted queue finds it
        last — the case that shows why random order is a floor, not a comparator.
        """
        y = _labels(0, 0, 1)
        amounts = pd.Series([900.0, 500.0, 10.0])

        report = ev.amount_sorted_baseline_report(y, amounts, top_k=(1, 2, 3))
        caught = dict(zip(report["top_k"], report["fraud_caught_by_amount"], strict=True))

        assert caught == {1: 0, 2: 0, 3: 1}

    def test_columns_are_renamed_for_side_by_side_merge(self):
        y = _labels(1, 0)
        amounts = pd.Series([10.0, 5.0])

        report = ev.amount_sorted_baseline_report(y, amounts, top_k=(1,))

        assert list(report.columns) == ["top_k", "fraud_caught_by_amount", "recall_by_amount"]
