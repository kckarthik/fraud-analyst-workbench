"""
App-layer guard for LLM-generated SQL. Defense in depth alongside the
read-only Postgres role (db/setup_readonly_role.sql) — even if a check here
had a gap, that role physically cannot write or see transactions.is_fraud.
"""
import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {"accounts", "analyst_transactions", "alerts", "dispositions", "rules"}
BLOCKED_FUNCS = {"pg_sleep", "pg_read_file", "dblink", "pg_terminate_backend", "lo_import", "lo_export"}
MAX_LIMIT = 200


class SQLGuardError(Exception):
    pass


def validate_and_cap(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except Exception as e:
        raise SQLGuardError(f"Could not parse SQL: {e}") from e

    if len(statements) != 1:
        raise SQLGuardError("Exactly one SQL statement is required.")

    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise SQLGuardError("Only SELECT statements are allowed.")

    forbidden_types = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter,
                        exp.Create, exp.Command, exp.Into)
    if any(stmt.find(t) for t in forbidden_types):
        raise SQLGuardError("Only read-only SELECT queries are allowed.")

    tables = {t.name.lower() for t in stmt.find_all(exp.Table)}
    disallowed = tables - ALLOWED_TABLES
    if disallowed:
        raise SQLGuardError(f"Query references disallowed table(s): {', '.join(disallowed)}")

    funcs = {f.name.lower() for f in stmt.find_all(exp.Anonymous)}
    blocked = funcs & BLOCKED_FUNCS
    if blocked:
        raise SQLGuardError(f"Query uses disallowed function(s): {', '.join(blocked)}")

    existing_limit = stmt.args.get("limit")
    if existing_limit is None:
        stmt = stmt.limit(MAX_LIMIT)
    else:
        try:
            n = int(str(existing_limit.expression.this))
            if n > MAX_LIMIT:
                stmt.set("limit", exp.Limit(expression=exp.Literal.number(MAX_LIMIT)))
        except Exception:
            stmt.set("limit", exp.Limit(expression=exp.Literal.number(MAX_LIMIT)))

    return stmt.sql(dialect="postgres")
