import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate import EvalItem, evaluate_one, summarize

DB_PATH = Path(__file__).resolve().parents[1] / "database" / "hospital.duckdb"
KNOWN_TABLES = {
    "departments", "rooms", "wards", "beds", "doctors", "nurses", "helpers",
    "patients", "bed_records", "room_records", "appointments",
    "medical_records", "staff_shifts", "surgery_records",
}


@pytest.fixture()
def con():
    if not DB_PATH.exists():
        pytest.skip("database not built")
    c = duckdb.connect(str(DB_PATH), read_only=True)
    yield c
    c.close()


def test_exact_match_true_for_identical_sql(con):
    sql = "SELECT COUNT(*) FROM doctors"
    item = EvalItem("q", sql, "SELECT   count(*)   from doctors", known_tables=KNOWN_TABLES)
    result = evaluate_one(item, con)
    assert result.exact_match
    assert result.execution_match


def test_execution_match_true_for_semantically_equivalent_sql(con):
    gold = "SELECT department_name FROM departments WHERE department_id = 101"
    pred = "SELECT dep.department_name FROM departments dep WHERE dep.department_id = 101"
    item = EvalItem("q", gold, pred, known_tables=KNOWN_TABLES)
    result = evaluate_one(item, con)
    assert not result.exact_match  # different alias/text
    assert result.execution_match  # but same result set


def test_execution_mismatch_for_wrong_predicate(con):
    gold = "SELECT COUNT(*) FROM doctors WHERE department_id = 101"
    pred = "SELECT COUNT(*) FROM doctors WHERE department_id = 102"
    item = EvalItem("q", gold, pred, known_tables=KNOWN_TABLES)
    result = evaluate_one(item, con)
    assert result.execution_match is False


def test_invalid_sql_marked_not_valid_and_no_execution_match(con):
    gold = "SELECT COUNT(*) FROM doctors"
    pred = "DROP TABLE doctors"
    item = EvalItem("q", gold, pred, known_tables=KNOWN_TABLES)
    result = evaluate_one(item, con)
    assert not result.valid_sql
    assert result.execution_match is None


def test_syntax_error_detected(con):
    item = EvalItem("q", "SELECT COUNT(*) FROM doctors", "SELECT FROM WHERE ???", known_tables=KNOWN_TABLES)
    result = evaluate_one(item, con)
    assert result.syntax_error


def test_table_and_column_accuracy_partial_overlap(con):
    gold = "SELECT first_name, last_name FROM doctors"
    pred = "SELECT first_name FROM doctors"
    item = EvalItem("q", gold, pred, known_tables=KNOWN_TABLES)
    result = evaluate_one(item, con)
    assert result.table_accuracy == 1.0
    assert 0.0 < result.column_accuracy < 1.0


def test_aggregation_accuracy_detects_mismatched_function(con):
    gold = "SELECT SUM(payment_amount) FROM appointments"
    pred = "SELECT AVG(payment_amount) FROM appointments"
    item = EvalItem("q", gold, pred, known_tables=KNOWN_TABLES)
    result = evaluate_one(item, con)
    assert result.aggregation_accuracy == 0.0


def test_summarize_produces_overall_and_per_difficulty(con):
    items = [
        EvalItem("q1", "SELECT COUNT(*) FROM doctors", "SELECT COUNT(*) FROM doctors", difficulty="easy", known_tables=KNOWN_TABLES),
        EvalItem("q2", "SELECT COUNT(*) FROM patients", "DROP TABLE patients", difficulty="hard", known_tables=KNOWN_TABLES),
    ]
    results = [evaluate_one(it, con) for it in items]
    report = summarize(results)
    assert report.n == 2
    assert "easy" in report.by_difficulty
    assert "hard" in report.by_difficulty
    assert report.by_difficulty["easy"]["exact_match_rate"] == 1.0
    assert report.by_difficulty["hard"]["valid_sql_rate"] == 0.0
