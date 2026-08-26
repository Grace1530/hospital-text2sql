"""
End-to-end SAFE query pipeline used by the web application:

    Natural-language question
        -> Text2SQLInferenceEngine (our Transformer, local, no external LLM)
        -> Generated SQL
        -> SQL cleanup
        -> SQL safety validator (src/sql/safety.py) -- READ-ONLY enforcement
        -> DuckDB (read-only connection)
        -> Results

Generated SQL is NEVER executed directly. It always passes through
validate_sql_safety() first, and the DuckDB connection used to execute it
is opened read_only=True as a second, independent layer of defense.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from src.inference.generate import Text2SQLInferenceEngine
from src.sql.safety import validate_sql_safety


@dataclass
class QueryResult:
    question: str
    generated_sql: str
    is_valid: bool
    validation_reason: str
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error: str | None = None


def clean_generated_sql(sql: str) -> str:
    """Light cleanup of raw model output before validation (trim, drop trailing junk)."""
    sql = sql.strip()
    # A model can emit multiple semicolon-separated fragments; keep only the first
    # statement -- validate_sql_safety would reject multiple statements anyway,
    # but trimming here gives a cleaner error/result for a common failure mode.
    if ";" in sql:
        sql = sql.split(";")[0].strip()
    return sql


class SafeQueryPipeline:
    def __init__(self, inference_engine: Text2SQLInferenceEngine, db_path: str | Path, schema_text: str):
        self.engine = inference_engine
        self.db_path = str(db_path)
        self.schema_text = schema_text
        con = duckdb.connect(self.db_path, read_only=True)
        self.known_tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        con.close()

    def run(self, question: str, row_limit: int = 500) -> QueryResult:
        t0 = time.perf_counter()
        raw_sql = self.engine.generate_sql(question, self.schema_text)
        sql = clean_generated_sql(raw_sql)

        validation = validate_sql_safety(sql, known_tables=self.known_tables)
        if not validation.is_safe:
            return QueryResult(
                question=question,
                generated_sql=sql,
                is_valid=False,
                validation_reason=validation.reason,
                execution_time_ms=(time.perf_counter() - t0) * 1000,
            )

        exec_sql = validation.normalized_sql or sql
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            cursor = con.execute(exec_sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(row_limit)
            return QueryResult(
                question=question,
                generated_sql=sql,
                is_valid=True,
                validation_reason="ok",
                columns=columns,
                rows=[list(r) for r in rows],
                row_count=len(rows),
                execution_time_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:  # noqa: BLE001
            return QueryResult(
                question=question,
                generated_sql=sql,
                is_valid=True,
                validation_reason="ok",
                error=str(e),
                execution_time_ms=(time.perf_counter() - t0) * 1000,
            )
        finally:
            con.close()
