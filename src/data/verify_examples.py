"""
Automatic SQL verification for generated Text-to-SQL examples (Phase 5).

Every example's SQL is:
1. Checked for read-only safety (src/sql/safety.py).
2. Actually EXECUTED against the real clean DuckDB database.
3. Kept only if it executes successfully; rejected examples are recorded
   with the reason (never silently dropped).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import duckdb

from src.data.hospital_nl2sql_generator import Example
from src.sql.safety import validate_sql_safety


@dataclass
class VerifiedExample:
    question: str
    sql: str
    category: str
    difficulty: str
    tables: list[str]
    result_row_count: int
    source: str = "hospital_generated"


@dataclass
class RejectedExample:
    question: str
    sql: str
    reason: str


def verify_examples(
    con: duckdb.DuckDBPyConnection,
    examples: list[Example],
    known_tables: set[str],
) -> tuple[list[VerifiedExample], list[RejectedExample]]:
    verified: list[VerifiedExample] = []
    rejected: list[RejectedExample] = []

    for ex in examples:
        safety = validate_sql_safety(ex.sql, known_tables=known_tables)
        if not safety.is_safe:
            rejected.append(RejectedExample(ex.question, ex.sql, f"safety: {safety.reason}"))
            continue
        try:
            result = con.execute(ex.sql).fetchall()
        except Exception as e:  # noqa: BLE001
            rejected.append(RejectedExample(ex.question, ex.sql, f"execution error: {e}"))
            continue
        verified.append(
            VerifiedExample(
                question=ex.question,
                sql=ex.sql,
                category=ex.category,
                difficulty=ex.difficulty,
                tables=ex.tables,
                result_row_count=len(result),
            )
        )
    return verified, rejected


def deduplicate(verified: list[VerifiedExample]) -> tuple[list[VerifiedExample], dict]:
    seen_questions = set()
    seen_sql = set()
    out = []
    dup_question_count = 0
    dup_sql_count = 0
    for ex in verified:
        q_key = ex.question.strip().lower()
        sql_key = " ".join(ex.sql.split()).lower()
        if q_key in seen_questions:
            dup_question_count += 1
            continue
        if sql_key in seen_sql:
            dup_sql_count += 1
            continue
        seen_questions.add(q_key)
        seen_sql.add(sql_key)
        out.append(ex)
    stats = {
        "input_count": len(verified),
        "output_count": len(out),
        "duplicate_questions_removed": dup_question_count,
        "duplicate_sql_removed": dup_sql_count,
    }
    return out, stats
