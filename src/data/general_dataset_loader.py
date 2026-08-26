"""
Phase 6/7 — Load, verify, and normalize the general-purpose Text-to-SQL
dataset (gretelai/synthetic_text_to_sql) into our combined training format.

Every example is verified the same rigorous way as the hospital-specific
examples: its schema + seed data are actually created in a fresh in-memory
DuckDB database, and the target SQL is actually executed against it. Only
examples that execute successfully are kept. This also handles dialect
normalization: the source dataset does not target one specific SQL
dialect, so we attempt DuckDB execution directly (DuckDB's SQL is broadly
ANSI-compatible) and fall back to an sqlglot dialect transpile only when
direct execution fails; if both fail, the example is excluded (never
silently "fixed" with an unreliable conversion).
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import sqlglot

from src.sql.safety import validate_sql_safety

ALLOWED_TASK_TYPES = {"data retrieval", "analytics and reporting"}

DIFFICULTY_MAP = {
    "basic SQL": "easy",
    "aggregation": "medium",
    "single join": "medium",
    "CTEs": "medium",
    "subqueries": "hard",
    "window functions": "hard",
    "multiple_joins": "hard",
    "set operations": "hard",
}


@dataclass
class GeneralExample:
    question: str
    sql: str
    schema_text: str
    category: str
    difficulty: str
    tables: list[str]
    domain: str
    result_row_count: int
    source: str = "gretel_synthetic_text_to_sql"


def _split_statements(sql_text: str) -> list[str]:
    try:
        parsed = sqlglot.parse(sql_text, read="sqlite")
        stmts = [p.sql(dialect="duckdb") for p in parsed if p is not None]
        if stmts:
            return stmts
    except Exception:  # noqa: BLE001
        pass
    # Fallback: naive split (dataset's context statements don't contain
    # semicolons inside string literals in practice).
    return [s.strip() for s in sql_text.split(";") if s.strip()]


def _build_scratch_db(context_sql: str) -> duckdb.DuckDBPyConnection | None:
    con = duckdb.connect(":memory:")
    statements = _split_statements(context_sql)
    if not statements:
        con.close()
        return None
    for stmt in statements:
        try:
            con.execute(stmt)
        except Exception:  # noqa: BLE001
            con.close()
            return None
    return con


def _schema_text_from_scratch_db(con: duckdb.DuckDBPyConnection) -> tuple[str, list[str]]:
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    blocks = []
    for tname in sorted(tables):
        cols = con.execute(f"PRAGMA table_info('{tname}')").fetchall()
        lines = [f"TABLE {tname}"] + [f"- {c[1]}" for c in cols]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), tables


def process_row(row: dict) -> tuple[GeneralExample | None, str]:
    """Returns (example_or_None, rejection_reason)."""
    if row.get("sql_task_type") not in ALLOWED_TASK_TYPES:
        return None, f"task_type_excluded:{row.get('sql_task_type')}"

    question = (row.get("sql_prompt") or "").strip()
    sql = (row.get("sql") or "").strip()
    context = (row.get("sql_context") or "").strip()
    if not question or not sql or not context:
        return None, "missing_field"

    con = _build_scratch_db(context)
    if con is None:
        return None, "schema_setup_failed"

    try:
        known_tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        safety = validate_sql_safety(sql, known_tables=known_tables)
        if not safety.is_safe:
            # One retry after transpiling target SQL through sqlglot -> duckdb,
            # in case only a dialect quirk (e.g. quoting) tripped the parser.
            try:
                transpiled = sqlglot.transpile(sql, read="sqlite", write="duckdb")[0]
            except Exception:  # noqa: BLE001
                return None, f"safety:{safety.reason}"
            safety = validate_sql_safety(transpiled, known_tables=known_tables)
            if not safety.is_safe:
                return None, f"safety:{safety.reason}"
            sql = transpiled

        try:
            result = con.execute(safety.normalized_sql or sql).fetchall()
        except Exception as e:  # noqa: BLE001
            return None, f"execution_error:{e}"

        schema_text, tables = _schema_text_from_scratch_db(con)
        difficulty = DIFFICULTY_MAP.get(row.get("sql_complexity", ""), "medium")

        example = GeneralExample(
            question=question,
            sql=safety.normalized_sql or sql,
            schema_text=schema_text,
            category=row.get("sql_complexity", "unknown"),
            difficulty=difficulty,
            tables=tables,
            domain=row.get("domain", ""),
            result_row_count=len(result),
        )
        return example, "ok"
    finally:
        con.close()
