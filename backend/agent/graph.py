"""
LangGraph NL2SQL agent: generate SQL -> validate -> execute -> summarize,
with bounded self-correction retries when validation or execution fails.
"""
import os
import sys
from typing import TypedDict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "db"))

from db_utils import get_readonly_engine
from langgraph.graph import END, StateGraph
from sqlalchemy import text

from .llm import generate
from .schema_context import SCHEMA_CONTEXT
from .sql_guard import SQLGuardError, validate_and_cap

MAX_ATTEMPTS = 3

SQL_SYSTEM = f"""You are a SQL generator for a fraud-analytics Postgres database.
{SCHEMA_CONTEXT}
Return ONLY the SQL query. No explanation, no markdown code fences, no commentary."""

ANSWER_SYSTEM = (
    "You are a fraud-analytics assistant. Answer using ONLY the given query "
    "results. Be concise (1-3 sentences) and cite concrete numbers.\n"
    "Describe the DATA, never the query. Do not mention SQL, table names, "
    "column names, or clauses such as LIMIT/GROUP BY. In particular the "
    "guard-injected LIMIT is not a finding — small models otherwise report "
    "'LIMIT 200' as though 200 rows were returned. Do not speculate beyond "
    "the rows you were given."
)


class AgentState(TypedDict):
    question: str
    sql: str
    error: str | None
    columns: list
    rows: list
    answer: str
    attempts: int


def _strip_code_fence(text_: str) -> str:
    t = text_.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("sql"):
            t = t[3:]
    return t.strip()


def generate_sql(state: AgentState) -> AgentState:
    feedback = f"\n\nThe previous attempt failed with: {state['error']}\nFix the query." if state.get("error") else ""
    prompt = f"Question: {state['question']}{feedback}\n\nSQL:"
    raw = generate(prompt, system=SQL_SYSTEM)
    return {**state, "sql": _strip_code_fence(raw), "attempts": state.get("attempts", 0) + 1}


def validate_sql(state: AgentState) -> AgentState:
    try:
        safe_sql = validate_and_cap(state["sql"])
        return {**state, "sql": safe_sql, "error": None}
    except SQLGuardError as e:
        return {**state, "error": str(e)}


def execute_sql(state: AgentState) -> AgentState:
    engine = get_readonly_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(state["sql"]))
            columns = list(result.keys())
            rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
        return {**state, "columns": columns, "rows": rows, "error": None}
    except Exception as e:
        return {**state, "error": str(e)}


def summarize(state: AgentState) -> AgentState:
    if state.get("error"):
        return {**state, "answer": f"I couldn't answer that after {state['attempts']} attempt(s): {state['error']}"}

    preview = state["rows"][:20]
    prompt = (
        f"Question: {state['question']}\n"
        f"SQL used: {state['sql']}\n"
        f"Result columns: {state['columns']}\n"
        f"Result rows (showing up to 20 of {len(state['rows'])} total): {preview}\n\n"
        "Answer the question in 1-3 plain-language sentences, citing concrete numbers from the results."
    )
    answer = generate(prompt, system=ANSWER_SYSTEM)
    return {**state, "answer": answer}


def route_after_validate(state: AgentState) -> str:
    if state.get("error"):
        return "retry" if state["attempts"] < MAX_ATTEMPTS else "give_up"
    return "execute"


def route_after_execute(state: AgentState) -> str:
    if state.get("error"):
        return "retry" if state["attempts"] < MAX_ATTEMPTS else "give_up"
    return "summarize"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("generate_sql", generate_sql)
    g.add_node("validate_sql", validate_sql)
    g.add_node("execute_sql", execute_sql)
    g.add_node("summarize", summarize)

    g.set_entry_point("generate_sql")
    g.add_edge("generate_sql", "validate_sql")
    g.add_conditional_edges(
        "validate_sql", route_after_validate,
        {"execute": "execute_sql", "retry": "generate_sql", "give_up": "summarize"},
    )
    g.add_conditional_edges(
        "execute_sql", route_after_execute,
        {"summarize": "summarize", "retry": "generate_sql", "give_up": "summarize"},
    )
    g.add_edge("summarize", END)
    return g.compile()


_graph = None


def run_agent(question: str) -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()
    init_state: AgentState = {
        "question": question, "sql": "", "error": None,
        "columns": [], "rows": [], "answer": "", "attempts": 0,
    }
    return _graph.invoke(init_state)
