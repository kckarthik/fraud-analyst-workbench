"""
Trains the Phase 2 LightGBM ranking model on the seeded dispositions and
reports business-facing queue-depth metrics on a held-out, chronologically
later slice of alerts.

Usage:
    python model/train.py
    python model/train.py --test-frac 0.2 --depths 0.05,0.10,0.20,0.30,0.50
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "db"))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from db_utils import get_engine
from evaluate import (
    amount_sorted_baseline_report,
    queue_depth_report,
    queue_depth_report_at_k,
    random_baseline_report,
    wilson_interval,
)
from features import build_feature_matrix
from sklearn.metrics import average_precision_score, roc_auc_score

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def temporal_train_test_split(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, test_frac: float = 0.2):
    """
    Train on the chronologically earlier alerts, test on the later ones —
    mirrors how a real deployment validates (score tomorrow's alerts with a
    model trained on yesterday's), not a random split which would leak
    future account behavior into training.
    """
    order = np.argsort(meta["triggered_at"].values, kind="mergesort")
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)

    cutoff = int(round(len(meta) * (1 - test_frac)))
    cutoff = min(max(cutoff, 1), len(meta) - 1)
    split_ts = meta["triggered_at"].iloc[cutoff]

    train = (X.iloc[:cutoff].reset_index(drop=True), y.iloc[:cutoff].reset_index(drop=True), meta.iloc[:cutoff].reset_index(drop=True))
    test = (X.iloc[cutoff:].reset_index(drop=True), y.iloc[cutoff:].reset_index(drop=True), meta.iloc[cutoff:].reset_index(drop=True))
    return train, test, split_ts


def train_model(X_train: pd.DataFrame, y_train: pd.Series, categorical_cols):
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    # Mild imbalance correction, NOT the full n_neg/n_pos ratio. The full ratio
    # (~2000:1 here) over-weights the positives so hard that every prediction
    # saturates to 0 or 1 — fine for a classifier, useless for a *ranking* model
    # because the scores collapse into two giant ties and can't order the queue.
    # A capped/sqrt weight keeps fraud sensitivity while leaving the scores spread
    # across (0, 1) so alerts can actually be ranked against each other.
    scale_pos_weight = min(np.sqrt(n_neg / n_pos), 25.0) if n_pos else 1.0

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_cols)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-frac", type=float, default=0.2,
                         help="Fraction of (chronologically latest) alerts held out as the test set")
    parser.add_argument("--depths", type=str, default="0.05,0.10,0.20,0.30,0.50",
                         help="Comma-separated queue-depth fractions to report precision/recall at")
    parser.add_argument("--top-k", type=str, default="100,250,500,1000,5000",
                         help="Comma-separated absolute queue depths (number of alerts reviewed) to report "
                              "recall at — the operationally meaningful cutoffs, since a team works a fixed "
                              "number of alerts per shift, not a fixed percentage of the queue")
    args = parser.parse_args()
    depths = tuple(float(d) for d in args.depths.split(","))
    top_k = tuple(int(k) for k in args.top_k.split(","))

    engine = get_engine()
    print("Building feature matrix from alerts.enrichment + raw transaction/account data ...")
    X, y, meta, categorical_cols = build_feature_matrix(engine)
    print(f"  {len(X):,} alerts, {y.sum():,} confirmed fraud ({y.mean():.2%})")

    (X_train, y_train, meta_train), (X_test, y_test, meta_test), split_ts = temporal_train_test_split(X, y, meta, args.test_frac)
    print(f"Temporal split at {split_ts}:")
    print(f"  train: {len(X_train):,} alerts, {meta_train['triggered_at'].min()} -> {meta_train['triggered_at'].max()}, "
          f"{y_train.sum()} fraud ({y_train.mean():.2%})")
    print(f"  test:  {len(X_test):,} alerts, {meta_test['triggered_at'].min()} -> {meta_test['triggered_at'].max()}, "
          f"{y_test.sum()} fraud ({y_test.mean():.2%})")

    if y_train.sum() == 0 or y_test.sum() == 0:
        print("WARNING: train or test split has zero confirmed-fraud examples — metrics below will be "
              "degenerate. This is a data-volume/time-span issue (re-run Phase 1 with a larger --sample "
              "or the full dataset), not a code issue.")

    print("Training LightGBM ...")
    model = train_model(X_train, y_train, categorical_cols)

    y_score = model.predict_proba(X_test)[:, 1]
    has_both_classes = y_test.nunique() > 1
    auc = roc_auc_score(y_test, y_score) if has_both_classes else float("nan")
    ap = average_precision_score(y_test, y_score) if has_both_classes else float("nan")
    print(f"\nTest ROC-AUC: {auc:.4f}  |  Test PR-AUC: {ap:.4f}")

    n_test_fraud = int(y_test.sum())
    if 0 < n_test_fraud < 100:
        lo, hi = wilson_interval(n_test_fraud - 1, n_test_fraud)
        print(f"\nNOTE: the test split contains only {n_test_fraud} confirmed-fraud alerts, so every recall "
              f"figure below is a small-sample estimate. One additional miss moves recall by "
              f"{100 / n_test_fraud:.1f} points, and missing a single alert gives a 95% interval of "
              f"[{lo:.1%}, {hi:.1%}]. Quote these as counts (\"{n_test_fraud - 1} of {n_test_fraud}\"), "
              f"not as three-significant-figure percentages.")

    report_at_k = queue_depth_report_at_k(y_test, y_score, top_k)
    amount_baseline = amount_sorted_baseline_report(y_test, X_test["sf_amount"], top_k)
    print("\nRecall at absolute queue depth — 'if the team works the top K alerts, how much fraud is caught?'")
    print("This is the operational number: a team works a few hundred alerts a shift, not a % of the queue.")
    print(report_at_k.merge(amount_baseline, on="top_k", how="left").to_string(index=False))

    report = queue_depth_report(y_test, y_score, depths)
    baseline = random_baseline_report(y_test, depths)
    print("\nQueue-depth precision/recall by percentage depth (retained for reference — on a queue this "
          "large a 5% cutoff is tens of thousands of alerts, so these rows overstate the review effort "
          "the ranking actually requires):")
    print(report.to_string(index=False))
    print("\nUnranked (random-order) baseline — a floor, not a realistic comparator; a team without a "
          "model sorts by amount (see the by-amount columns above), it does not work at random:")
    print(baseline.to_string(index=False))

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "lgbm_model.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "feature_columns.json"), "w") as f:
        json.dump({"columns": list(X.columns), "categorical": categorical_cols}, f, indent=2)

    metrics = {
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "split_timestamp": str(split_ts),
        "roc_auc": auc,
        "pr_auc": ap,
        "n_test_fraud": n_test_fraud,
        "queue_depth_report_at_k": report_at_k.to_dict(orient="records"),
        "amount_sorted_baseline_at_k": amount_baseline.to_dict(orient="records"),
        "queue_depth_report": report.to_dict(orient="records"),
        "random_baseline_report": baseline.to_dict(orient="records"),
    }
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved model + feature schema + metrics to {ARTIFACT_DIR}/")
    print("Next: python model/explain.py  (scores all alerts + writes SHAP reason codes into alerts.enrichment)")


if __name__ == "__main__":
    main()
