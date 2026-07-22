"""
Feature construction for real-time scoring (POST /api/score).

The offline pipeline scores a table it already has: rules, enrichment and the
model all run over the full transaction history at once. Online, a transaction
arrives on its own and every history-dependent feature — 24h velocity, amount
z-score against the account's own baseline, prior alert counts, and half the
rules — has to be reconstructed from whatever the database knows about that
account at that instant.

The approach here is deliberate: fetch the account's prior transactions, append
the candidate, and run the *same* rule and enrichment functions the training
pipeline used over that small per-account frame, taking the last row. Nothing is
reimplemented. Every rule in rules/rule_definitions.py is already a pure
function of a DataFrame, so a two-row frame is as valid an input as a
million-row one, and reuse is what keeps the online features identical to the
ones the model was trained on.

Known and intentional limits, since honesty about them matters more than
pretending they aren't there:

  * The candidate is scored against history *already committed to the database*.
    Two transactions arriving within the same instant will not see each other.
    A real deployment puts a streaming feature store here; this reads Postgres.
  * prior_alert_count / prior_fp_rate read the dispositions table, which in this
    build is seeded from ground truth (see README). Offline training used the
    same values, so the two paths agree — but on real data these would only be
    populated by genuine analyst review.
  * History is capped (HISTORY_LIMIT). An account with more prior transactions
    than that gets its most recent ones, which is enough for every window the
    features actually use (24h velocity, same-day rules) but makes the expanding
    z-score baseline approximate for extremely active accounts.
"""
import importlib.util
import json
import os
import sys

import pandas as pd
from sqlalchemy import text

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT_DIR = os.path.join(ROOT, "model", "artifacts")

# Rules needing device/identity/region fields PaySim does not carry. Must match
# the --skip-rules passed to rules/engine.py, or the online rule dummies will
# disagree with the ones the model trained on.
SKIPPED_RULES = {"new_device", "missing_identity_high_amount", "region_mismatch"}

# Upper bound on prior transactions pulled per request.
HISTORY_LIMIT = 500


def _load(module_name: str, relative_path: str):
    """
    Import by explicit path under a distinct name.

    model/features.py and enrichment/features.py share a basename, so a plain
    sys.path insert would make which one you get depend on ordering — the same
    hazard tests/conftest.py works around, handled the same way.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(ROOT, relative_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


model_features = _load("model_features", "model/features.py")
enrichment_features = _load("enrichment_features", "enrichment/features.py")
rule_definitions = _load("rule_definitions", "rules/rule_definitions.py")
reason_codes = _load("model_reason_codes", "model/reason_codes.py")


class UnknownAccountError(Exception):
    """Raised when the transaction references an account with no record."""


def _feature_schema() -> dict:
    with open(os.path.join(ARTIFACT_DIR, "feature_columns.json")) as f:
        return json.load(f)


def load_account_context(conn, account_id: str, ts: pd.Timestamp) -> tuple[dict, pd.DataFrame, dict]:
    """
    Everything the database knows about this account strictly before `ts`:
    the account record, its prior transactions, and its prior alert history.
    """
    account = conn.execute(
        text("SELECT account_id, first_seen_at, account_type, card_network, region_code "
             "FROM accounts WHERE account_id = :aid"),
        {"aid": account_id},
    ).mappings().first()
    if account is None:
        raise UnknownAccountError(account_id)

    # Strictly-prior only: a transaction must never see itself, nor anything
    # recorded after it, or the online score silently uses information the
    # offline features could not have had.
    #
    # This does mean a transaction sharing the candidate's exact timestamp is
    # invisible here, which is the right call — at equal timestamps nothing in
    # the data establishes which happened first — but it is the one remaining
    # source of batch/online disagreement, since the batch sort has to break
    # those ties somehow. PaySim's clock is hourly, so ties are real (9 accounts
    # in this dataset). Resolving it properly needs a monotonic sequence number
    # or sub-second precision on the transaction, not a rule invented here.
    history = pd.read_sql(
        text("""
            SELECT transaction_id, account_id, ts, amount, transaction_type,
                   counterparty_id, counterparty_region, device_id, has_identity_data,
                   orig_balance_before, orig_balance_after,
                   dest_balance_before, dest_balance_after
            FROM transactions
            WHERE account_id = :aid AND ts < :ts
            ORDER BY ts DESC
            LIMIT :lim
        """),
        conn,
        params={"aid": account_id, "ts": ts, "lim": HISTORY_LIMIT},
    )

    alert_stats = conn.execute(
        text("""
            SELECT COUNT(*) AS prior_alert_count,
                   COUNT(*) FILTER (WHERE d.decision = 'not_fraud') AS prior_fp_count
            FROM alerts a
            LEFT JOIN dispositions d ON d.alert_id = a.alert_id
            WHERE a.account_id = :aid AND a.triggered_at < :ts
        """),
        {"aid": account_id, "ts": ts},
    ).mappings().first()

    return dict(account), history, dict(alert_stats)


def _candidate_row(txn: dict, account: dict) -> dict:
    return {
        # Synthetic id: the rules need something to count, and this transaction
        # is not in the table yet.
        "transaction_id": "__candidate__",
        "account_id": txn["account_id"],
        "ts": txn["ts"],
        "amount": float(txn["amount"]),
        "transaction_type": txn["transaction_type"],
        "counterparty_id": txn.get("counterparty_id"),
        "counterparty_region": txn.get("counterparty_region"),
        "device_id": txn.get("device_id"),
        "has_identity_data": bool(txn.get("has_identity_data", False)),
        "orig_balance_before": txn.get("orig_balance_before"),
        "orig_balance_after": txn.get("orig_balance_after"),
        "dest_balance_before": txn.get("dest_balance_before"),
        "dest_balance_after": txn.get("dest_balance_after"),
    }


def _run_rules(frame: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate the rule registry over the account's frame. Same functions, same
    registry, same skip list as rules/engine.py — a rule reimplemented here
    would be a rule that fires differently online than the model was taught.
    """
    fired = pd.DataFrame(index=frame.index)
    for rule_id, (fn, _desc, kwargs) in rule_definitions.RULE_REGISTRY.items():
        if rule_id in SKIPPED_RULES:
            fired[rule_id] = False
            continue
        fired[rule_id] = fn(frame, accounts) if kwargs.get("needs_accounts") else fn(frame)
    return fired


def build_scoring_row(conn, txn: dict) -> tuple[pd.DataFrame, list[str], dict, int]:
    """
    Returns (X, fired_rule_ids, structured_facts, n_history) for one candidate
    transaction.

    X is a single-row frame whose columns and dtypes match the trained model's
    schema exactly, taken from the saved feature_columns.json rather than
    rebuilt by hand. n_history is how many prior transactions the
    history-dependent features were actually computed from.
    """
    ts = pd.Timestamp(txn["ts"])
    account, history, alert_stats = load_account_context(conn, txn["account_id"], ts)

    candidate = _candidate_row(txn, account)
    candidate_df = pd.DataFrame([candidate])
    # Concatenating onto an empty history makes pandas infer dtypes from all-NA
    # columns and warn; a first-ever transaction is just the candidate alone.
    frame = candidate_df if history.empty else pd.concat([history, candidate_df], ignore_index=True)
    frame["ts"] = pd.to_datetime(frame["ts"])
    # Explicit per-value coercion rather than fillna().astype(bool): the column
    # arrives as object dtype when history is empty, and fillna on object dtype
    # silently downcasts (deprecated, and version-dependent).
    frame["has_identity_data"] = frame["has_identity_data"].map(
        lambda v: bool(v) if pd.notna(v) else False
    ).astype(bool)
    # (account_id, ts) order is what every rule and the velocity features assume.
    frame = frame.sort_values(["account_id", "ts"]).reset_index(drop=True)

    accounts_df = pd.DataFrame([{
        "account_id": account["account_id"],
        "first_seen_at": pd.Timestamp(account["first_seen_at"]),
        "account_type": account["account_type"],
        "card_network": account["card_network"],
        "region_code": account["region_code"],
    }])

    fired = _run_rules(frame, accounts_df)
    enriched = enrichment_features.compute_velocity_features(frame)
    enriched["amount_zscore"] = enrichment_features.compute_amount_zscore(enriched)

    # The candidate is identified by its synthetic id, never by position: the
    # sort above interleaves it with history by timestamp, so "the last row" is
    # only the candidate when it happens to be the newest transaction. A
    # backdated transaction would silently score a different row.
    is_candidate = enriched["transaction_id"] == "__candidate__"
    if not is_candidate.any():
        raise RuntimeError("candidate row lost during feature construction")
    row = enriched[is_candidate].iloc[[0]].reset_index(drop=True)
    fired_row = fired[(frame["transaction_id"] == "__candidate__").to_numpy()].reset_index(drop=True)
    fired_rule_ids = [r for r in fired_row.columns if bool(fired_row[r].iat[0])]

    prior_alert_count = int(alert_stats["prior_alert_count"] or 0)
    prior_fp_count = int(alert_stats["prior_fp_count"] or 0)
    structured_facts = {
        "amount": float(row["amount"].iat[0]),
        "amount_zscore": round(float(row["amount_zscore"].iat[0]), 2),
        "velocity_24h_count": int(row["velocity_24h_count"].iat[0]),
        "velocity_24h_sum": round(float(row["velocity_24h_sum"].iat[0]), 2),
        "prior_alert_count": prior_alert_count,
        "prior_fp_rate": round(prior_fp_count / prior_alert_count, 3) if prior_alert_count else 0.0,
        "rule_ids": fired_rule_ids,
    }

    # Shape the row exactly as model/features.py does offline, via the shared
    # derive_row_features/assemble_X rather than a parallel implementation.
    raw = row.copy()
    raw["first_seen_at"] = pd.Timestamp(account["first_seen_at"])
    raw["account_type"] = account["account_type"]
    raw["card_network"] = account["card_network"]
    raw["account_region"] = account["region_code"]
    raw = model_features.derive_row_features(raw, event_time_col="ts")

    sf = pd.DataFrame([{f"sf_{k}": structured_facts[k] for k in model_features.SF_FIELDS}])
    rule_dummies = model_features._rule_dummies(pd.Series([fired_rule_ids]))

    X = model_features.assemble_X(raw, sf, rule_dummies)

    # Reindex to the trained column order. Guards against a feature added to
    # one path and not the other, which LightGBM would otherwise accept
    # positionally and score against the wrong column.
    schema = _feature_schema()
    missing = [c for c in schema["columns"] if c not in X.columns]
    if missing:
        raise RuntimeError(f"online features missing trained columns: {missing}")
    X = X[schema["columns"]]

    return X, fired_rule_ids, structured_facts, len(history)
