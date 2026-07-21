"""
Security tests for the LLM-generated-SQL guard.

This is the highest-risk code in the repo: an LLM writes SQL and we execute it.
The guard is defense in depth alongside the least-privilege Postgres role, but
each layer has to hold on its own — these tests cover the guard in isolation,
assuming the role does not exist.

`transactions` is deliberately absent from the allow-list: it holds `is_fraud`,
the ground-truth label. Reaching it via any route (direct, CTE, subquery, join,
UNION) is the single most important thing to block, so those get their own
tests rather than one representative case.
"""
import pytest
import sqlglot
from agent.sql_guard import MAX_LIMIT, SQLGuardError, validate_and_cap


# --------------------------------------------------------------------------
# Writes and DDL
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO dispositions (alert_id, decision) VALUES (1, 'fraud')",
        "UPDATE alerts SET status = 'closed'",
        "DELETE FROM alerts WHERE alert_id = 1",
        "DROP TABLE alerts",
        "ALTER TABLE alerts ADD COLUMN x INT",
        "CREATE TABLE evil (id INT)",
        "TRUNCATE alerts",
        "GRANT ALL ON alerts TO PUBLIC",
    ],
)
def test_rejects_writes_and_ddl(sql):
    with pytest.raises(SQLGuardError):
        validate_and_cap(sql)


def test_rejects_select_into():
    """SELECT ... INTO creates a table — a write wearing a SELECT's clothes."""
    with pytest.raises(SQLGuardError):
        validate_and_cap("SELECT * INTO evil FROM alerts")


# --------------------------------------------------------------------------
# Statement stacking
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 FROM alerts; DROP TABLE alerts",
        "SELECT 1 FROM alerts; DELETE FROM dispositions",
        "SELECT 1 FROM alerts;SELECT 2 FROM alerts",
    ],
)
def test_rejects_stacked_statements(sql):
    with pytest.raises(SQLGuardError):
        validate_and_cap(sql)


def test_rejects_statement_stacked_behind_a_comment():
    """
    A regex-based guard scanning for a leading SELECT is defeated by this.
    Parsing is what makes it safe: sqlglot sees two statements regardless of
    the comment.
    """
    with pytest.raises(SQLGuardError):
        validate_and_cap("SELECT 1 FROM alerts -- harmless\n; DROP TABLE alerts")


# --------------------------------------------------------------------------
# Reaching the hidden ground-truth table
# --------------------------------------------------------------------------
def test_rejects_direct_access_to_transactions():
    with pytest.raises(SQLGuardError):
        validate_and_cap("SELECT is_fraud FROM transactions")


def test_rejects_transactions_via_cte():
    with pytest.raises(SQLGuardError):
        validate_and_cap(
            "WITH leak AS (SELECT is_fraud FROM transactions) SELECT * FROM leak"
        )


def test_rejects_transactions_via_subquery():
    with pytest.raises(SQLGuardError):
        validate_and_cap(
            "SELECT (SELECT COUNT(*) FROM transactions WHERE is_fraud) AS n FROM alerts"
        )


def test_rejects_transactions_via_join():
    with pytest.raises(SQLGuardError):
        validate_and_cap(
            "SELECT a.alert_id, t.is_fraud FROM alerts a "
            "JOIN transactions t ON t.transaction_id = a.transaction_id"
        )


def test_rejects_transactions_via_union():
    with pytest.raises(SQLGuardError):
        validate_and_cap(
            "SELECT alert_id FROM alerts UNION SELECT is_fraud FROM transactions"
        )


def test_rejects_unknown_table():
    with pytest.raises(SQLGuardError):
        validate_and_cap("SELECT * FROM pg_shadow")


# --------------------------------------------------------------------------
# Dangerous functions
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_sleep(10) FROM alerts",
        "SELECT pg_read_file('/etc/passwd') FROM alerts",
        "SELECT lo_import('/etc/passwd') FROM alerts",
    ],
)
def test_rejects_blocked_functions(sql):
    with pytest.raises(SQLGuardError):
        validate_and_cap(sql)


# --------------------------------------------------------------------------
# Malformed input
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sql", ["", "   ", "not sql at all", "SELECT FROM WHERE"])
def test_rejects_unparseable_input(sql):
    with pytest.raises(SQLGuardError):
        validate_and_cap(sql)


# --------------------------------------------------------------------------
# LIMIT enforcement
# --------------------------------------------------------------------------
def _limit_of(sql: str) -> int:
    parsed = sqlglot.parse_one(sql, read="postgres")
    return int(parsed.args["limit"].expression.this)


def test_injects_limit_when_absent():
    assert _limit_of(validate_and_cap("SELECT alert_id FROM alerts")) == MAX_LIMIT


def test_caps_limit_above_maximum():
    assert _limit_of(validate_and_cap("SELECT alert_id FROM alerts LIMIT 100000")) == MAX_LIMIT


def test_preserves_limit_below_maximum():
    assert _limit_of(validate_and_cap("SELECT alert_id FROM alerts LIMIT 5")) == 5


# --------------------------------------------------------------------------
# Legitimate queries must survive
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM analyst_dispositions WHERE decision = 'fraud'",
        "SELECT alert_id, model_score FROM alerts ORDER BY model_score DESC",
        "SELECT at.transaction_type, COUNT(*) FROM analyst_dispositions d "
        "JOIN alerts a ON a.alert_id = d.alert_id "
        "JOIN analyst_transactions at ON at.transaction_id = a.transaction_id "
        "WHERE d.decision = 'fraud' GROUP BY at.transaction_type",
        "SELECT r.rule_name, COUNT(*) FROM rules r GROUP BY r.rule_name",
        "SELECT account_id FROM accounts LIMIT 10",
    ],
)
def test_allows_legitimate_queries(sql):
    assert validate_and_cap(sql).strip()


# --------------------------------------------------------------------------
# The raw dispositions table is the answer key
# --------------------------------------------------------------------------
# rules/engine.py backfills a disposition for every alert straight from the
# dataset's is_fraud label. Blocking transactions.is_fraud while leaving that
# table reachable is theatre — the label is simply read from the other copy,
# which is exactly what the agent did in practice. These pin the block on every
# route the transactions tests already cover.
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM dispositions WHERE decision = 'fraud'",
        "SELECT d.decision FROM dispositions d WHERE d.alert_id = 1",
        "WITH x AS (SELECT decision FROM dispositions) SELECT * FROM x",
        "SELECT (SELECT decision FROM dispositions WHERE alert_id = a.alert_id) FROM alerts a",
        "SELECT a.alert_id FROM alerts a JOIN dispositions d ON d.alert_id = a.alert_id",
        "SELECT decision FROM analyst_dispositions UNION ALL SELECT decision FROM dispositions",
    ],
)
def test_blocks_raw_dispositions_table(sql):
    with pytest.raises(SQLGuardError):
        validate_and_cap(sql)
