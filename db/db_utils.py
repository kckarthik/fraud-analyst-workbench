"""
Shared database connection + bulk-write helpers.
Reads connection details from .env (copy .env.example to .env first).
"""
import csv
import io
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required connection configuration is missing."""


def _require(var: str) -> str:
    """
    Fetch a required env var, failing loudly if it's absent.

    Deliberately no default: a fallback password in source is how a dev
    credential ends up silently working in a non-dev environment. Missing
    config should stop the process, not quietly connect as something else.
    """
    value = os.getenv(var)
    if not value:
        raise ConfigError(
            f"Required environment variable {var} is not set. "
            "Copy .env.example to .env and fill in local values."
        )
    return value


def _build_url(user: str, password: str) -> str:
    # quote_plus: passwords containing @ / : / # would otherwise corrupt the
    # URL and produce a confusing "could not translate host name" error.
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "fraud_intel")
    return (
        f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{name}"
    )


def get_engine():
    return create_engine(_build_url(_require("DB_USER"), _require("DB_PASSWORD")))


def get_readonly_engine():
    """
    SELECT-only connection for the NL2SQL agent — a separate least-privilege
    Postgres role (see db/setup_readonly_role.sql), not the app's normal role.
    Defense in depth alongside the app-layer SQL guard: even if the guard were
    bypassed, this role physically cannot write or see transactions.is_fraud.
    """
    return create_engine(
        _build_url(_require("DB_READONLY_USER"), _require("DB_READONLY_PASSWORD"))
    )


def bulk_copy_insert(engine, df, table: str, columns: list, chunk_size: int = 500_000):
    """
    Fast bulk INSERT via Postgres COPY FROM STDIN instead of row-by-row/multi-value
    INSERT — orders of magnitude faster at millions of rows, since COPY skips
    per-statement parsing/planning overhead entirely. Chunked to bound peak
    memory rather than building one giant in-memory buffer.

    Any TEXT[]/array columns must already be pre-serialized to Postgres array
    literal strings (see pg_text_array) before being passed in.
    """
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cols_sql = ", ".join(columns)
            for start in tqdm(range(0, len(df), chunk_size)):
                chunk = df.iloc[start:start + chunk_size]
                buf = io.StringIO()
                # QUOTE_NONE: COPY's default TEXT format has no concept of CSV-style
                # quoting — a literal `"` (e.g. inside a serialized Postgres array
                # literal like {"rule_a"}) must pass through unescaped, not get
                # wrapped/doubled the way pandas' default CSV quoting would do.
                chunk.to_csv(buf, sep="\t", na_rep="\\N", index=False, header=False, columns=columns,
                             quoting=csv.QUOTE_NONE)
                buf.seek(0)
                cur.copy_expert(f"COPY {table} ({cols_sql}) FROM STDIN", buf)
        raw_conn.commit()
    finally:
        raw_conn.close()


def pg_text_array(values) -> str:
    """Serialize a Python list of strings to a Postgres TEXT[] array literal, for use with bulk_copy_insert."""
    if not values:
        return "{}"
    escaped = [str(v).replace("\\", "\\\\").replace('"', '\\"') for v in values]
    return "{" + ",".join(f'"{e}"' for e in escaped) + "}"


def bulk_jsonb_update(engine, records, table: str, key_col: str, json_col: str,
                      mode: str = "overwrite"):
    """
    Fast bulk update of a JSONB column keyed by an integer PK.

    `records` is an iterable of (key:int, json_string:str) pairs. Instead of
    thousands of per-chunk `UPDATE ... FROM (VALUES ...)` round trips (which cap
    out around a couple thousand rows/sec), this COPYs the payloads into a TEMP
    table once and runs a single indexed `UPDATE ... FROM` join — typically an
    order of magnitude faster on large batches.

    mode:
      "overwrite" -> json_col = payload
      "merge"     -> json_col = json_col || payload  (preserves existing keys)
    """
    if mode not in ("overwrite", "merge"):
        raise ValueError(f"mode must be 'overwrite' or 'merge', got {mode!r}")

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t")
    for key, payload in records:
        writer.writerow([int(key), payload])
    buf.seek(0)

    set_expr = "u.payload" if mode == "overwrite" else f"t.{json_col} || u.payload"
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE _bulk_jsonb_upd (key_val bigint, payload jsonb) ON COMMIT DROP")
            # CSV format so payloads containing quotes/tabs/newlines are escaped
            # correctly; COPY feeds each field to the jsonb input function.
            cur.copy_expert(
                "COPY _bulk_jsonb_upd (key_val, payload) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t')",
                buf,
            )
            cur.execute("CREATE INDEX ON _bulk_jsonb_upd (key_val)")
            cur.execute("ANALYZE _bulk_jsonb_upd")
            cur.execute(
                f"UPDATE {table} AS t SET {json_col} = {set_expr} "
                f"FROM _bulk_jsonb_upd u WHERE t.{key_col} = u.key_val"
            )
        raw_conn.commit()
    finally:
        raw_conn.close()
