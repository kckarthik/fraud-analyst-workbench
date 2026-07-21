"""
Business-facing evaluation for the ranking model: how much confirmed fraud is
caught if analysts work the queue in descending model-score order and stop at
some cutoff.

Two views of the same curve:
  - queue_depth_report      — cutoffs as a *fraction* of the queue
  - queue_depth_report_at_k — cutoffs as an *absolute* number of alerts

The absolute view is the operationally meaningful one. A percentage depth only
means something on a queue whose size matches a team's capacity; at ~1M alerts,
"top 5%" is 49,000 alerts — months of work for one analyst, so a 5% row says
nothing about whether the ranking is usable. Real teams work a few hundred
alerts a shift, so recall@500 is the number that maps onto a working day.
"""
import numpy as np
import pandas as pd

DEFAULT_DEPTHS = (0.05, 0.10, 0.20, 0.30, 0.50)
DEFAULT_TOP_K = (100, 250, 500, 1000, 5000)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion — the confidence band on a
    recall figure computed from n positives.

    Reported because recall here rests on a small number of confirmed-fraud
    alerts (tens, not thousands), where the normal approximation is badly
    behaved near p=1 and a bare point estimate implies far more precision than
    the sample supports. 68/69 is "somewhere around 92-99.7%", not "98.6%".
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _catch_curve(y_true: pd.Series, y_score: np.ndarray):
    """
    Cumulative confirmed-fraud count as the queue is worked in descending score
    order. Returns (cumulative_caught, total_fraud, n).
    """
    y = np.asarray(y_true)
    # mergesort => stable, so score ties keep their incoming (alert_id) order
    # rather than one that depends on the sort implementation. Ties are common
    # at the top of this queue, so an unstable sort makes the report irreproducible.
    order = np.argsort(-np.asarray(y_score), kind="mergesort")
    return np.cumsum(y[order]), int(y.sum()), len(y)


def queue_depth_report(y_true: pd.Series, y_score: np.ndarray, depths=DEFAULT_DEPTHS) -> pd.DataFrame:
    """Precision/recall if alerts are worked in descending model-score order, cut off at each fractional depth."""
    cum_caught, total_fraud, n = _catch_curve(y_true, y_score)

    rows = []
    for depth in depths:
        k = max(1, min(n, int(round(depth * n))))
        caught = int(cum_caught[k - 1])
        rows.append({
            "queue_depth_pct": round(depth * 100, 1),
            "alerts_reviewed": k,
            "fraud_caught": caught,
            "fraud_missed": total_fraud - caught,
            "total_fraud_in_test": total_fraud,
            "precision_at_depth": round(caught / k, 4),
            "recall_at_depth": round(caught / total_fraud, 4) if total_fraud else None,
        })
    return pd.DataFrame(rows)


def queue_depth_report_at_k(y_true: pd.Series, y_score: np.ndarray, top_k=DEFAULT_TOP_K) -> pd.DataFrame:
    """
    Recall at absolute queue depths — "if the team works the top K alerts, how
    much fraud do they catch?"

    Carries the random-order expectation inline rather than as a separate table:
    at these depths the expected catch from working an unranked queue is a
    fraction of one alert, which makes the comparison self-evident without
    quoting a lift multiple. Recall is reported with a Wilson interval because
    the denominator is the number of confirmed-fraud alerts in the test split,
    which is small.
    """
    cum_caught, total_fraud, n = _catch_curve(y_true, y_score)
    base_rate = total_fraud / n if n else 0.0

    rows = []
    for k_requested in top_k:
        # A cutoff deeper than the queue is the whole queue; skip it rather than
        # emitting a duplicate row for every K past the end.
        if k_requested > n:
            continue
        k = max(1, int(k_requested))
        caught = int(cum_caught[k - 1])
        ci_low, ci_high = wilson_interval(caught, total_fraud)
        rows.append({
            "top_k": k,
            "pct_of_queue": round(100 * k / n, 3),
            "fraud_caught": caught,
            "fraud_missed": total_fraud - caught,
            "total_fraud_in_test": total_fraud,
            "precision_at_k": round(caught / k, 4),
            "recall_at_k": round(caught / total_fraud, 4) if total_fraud else None,
            "recall_ci_low": round(ci_low, 4) if total_fraud else None,
            "recall_ci_high": round(ci_high, 4) if total_fraud else None,
            "random_expected_caught": round(base_rate * k, 2),
        })
    return pd.DataFrame(rows)


def random_baseline_report(y_true: pd.Series, depths=DEFAULT_DEPTHS) -> pd.DataFrame:
    """
    Expected catch rate working alerts in arbitrary (unranked) order.

    This is a floor, not a realistic comparator: a team with no model does not
    work the queue at random, it sorts by amount or by rule severity. Read it as
    "the ranking beats no ordering at all", and treat any lift multiple derived
    from it as an upper bound on the real-world gain.
    """
    n = len(y_true)
    total_fraud = int(np.asarray(y_true).sum())
    base_rate = total_fraud / n if n else 0.0

    rows = []
    for depth in depths:
        k = max(1, min(n, int(round(depth * n))))
        rows.append({
            "queue_depth_pct": round(depth * 100, 1),
            "alerts_reviewed": k,
            "expected_fraud_caught": round(base_rate * k, 1),
            "expected_recall_at_depth": round(depth, 4),
        })
    return pd.DataFrame(rows)


def amount_sorted_baseline_report(y_true: pd.Series, amounts: pd.Series, top_k=DEFAULT_TOP_K) -> pd.DataFrame:
    """
    Recall at absolute depths when the queue is sorted by transaction amount
    descending — what a team without a model actually does. A more honest
    comparator than random order, and the one a skeptical reader will ask for.
    """
    return queue_depth_report_at_k(y_true, np.asarray(amounts, dtype=float), top_k)[
        ["top_k", "fraud_caught", "recall_at_k"]
    ].rename(columns={"fraud_caught": "fraud_caught_by_amount", "recall_at_k": "recall_by_amount"})
