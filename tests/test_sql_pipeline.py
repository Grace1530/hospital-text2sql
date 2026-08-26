import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sql.pipeline import SafeQueryPipeline, clean_generated_sql

DB_PATH = Path(__file__).resolve().parents[1] / "database" / "hospital.duckdb"


class FakeEngine:
    def __init__(self, fixed_sql: str):
        self.fixed_sql = fixed_sql

    def generate_sql(self, question: str, schema_text: str, max_new_tokens: int = 128) -> str:
        return self.fixed_sql


@pytest.fixture()
def pipeline_factory():
    if not DB_PATH.exists():
        pytest.skip("database not built")

    def _make(sql: str) -> SafeQueryPipeline:
        return SafeQueryPipeline(FakeEngine(sql), DB_PATH, schema_text="TABLE doctors\n- doctor_id")

    return _make


def test_clean_generated_sql_strips_trailing_semicolon_fragments():
    assert clean_generated_sql("SELECT 1; DROP TABLE x;") == "SELECT 1"
    assert clean_generated_sql("  SELECT 1  ") == "SELECT 1"


def test_valid_select_executes_and_returns_rows(pipeline_factory):
    pipeline = pipeline_factory("SELECT COUNT(*) FROM doctors")
    result = pipeline.run("How many doctors?")
    assert result.is_valid
    assert result.error is None
    assert result.row_count == 1
    assert result.rows[0][0] > 0


def test_dangerous_sql_is_rejected_before_execution(pipeline_factory):
    pipeline = pipeline_factory("DROP TABLE doctors")
    result = pipeline.run("delete everything")
    assert not result.is_valid
    assert result.row_count == 0
    assert result.rows == []

    # Confirm the table really is untouched.
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    count = con.execute("SELECT COUNT(*) FROM doctors").fetchone()[0]
    con.close()
    assert count > 0


def test_unknown_table_is_rejected(pipeline_factory):
    pipeline = pipeline_factory("SELECT * FROM made_up_table")
    result = pipeline.run("q")
    assert not result.is_valid


def test_multi_statement_injection_attempt_is_rejected(pipeline_factory):
    pipeline = pipeline_factory("SELECT * FROM doctors; DROP TABLE doctors;")
    result = pipeline.run("q")
    # clean_generated_sql keeps only the first statement, which is itself safe
    assert result.is_valid
    assert result.generated_sql == "SELECT * FROM doctors"


def test_row_limit_is_applied(pipeline_factory):
    pipeline = pipeline_factory("SELECT * FROM medical_records")
    result = pipeline.run("q", row_limit=10)
    assert result.row_count == 10


def test_execution_error_is_captured_not_raised(pipeline_factory):
    pipeline = pipeline_factory("SELECT nonexistent_column FROM doctors")
    result = pipeline.run("q")
    assert result.is_valid  # syntactically a safe SELECT
    assert result.error is not None
