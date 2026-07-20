"""
Business-facing evaluation for the ranking model: precision/recall at
different alert-queue-depth thresholds — "if analysts only work the top N%
of ranked alerts, how much confirmed fraud do we still catch."
"""
import numpy as np
import pandas as pd

DEFAULT_DEPTHS = (0.05, 0.10, 0.20, 0.30, 0.50)


def queue_depth_report(y_true: pd.Series, y_score: np.ndarray, depths=DEFAULT_DEPTHS) -> pd.DataFrame:
    """Precision/recall if alerts are worked in descending model-score order, cut off at each depth."""
    n = len(y_true)
    total_fraud = int(np.asarray(y_true).sum())
    order = np.argsort(-np.asarray(y_score))
    y_sorted = np.asarray(y_true)[order]
    cum_fraud_caught = np.cumsum(y_sorted)

    rows = []
    for depth in depths:
        k = max(1, min(n, int(round(depth * n))))
        caught = int(cum_fraud_caught[k - 1])
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


def random_baseline_report(y_true: pd.Series, depths=DEFAULT_DEPTHS) -> pd.DataFrame:
    """
    Expected catch rate working alerts in arbitrary (unranked) order — the
    implicit baseline today, since Phase 1 alerts carry no priority signal.
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
