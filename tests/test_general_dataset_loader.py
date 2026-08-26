import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.general_dataset_loader import process_row

GOOD_ROW = {
    "sql_task_type": "analytics and reporting",
    "sql_complexity": "single join",
    "sql_prompt": "What is the total volume of timber sold by each salesperson?",
    "sql_context": (
        "CREATE TABLE salesperson (salesperson_id INT, name TEXT); "
        "INSERT INTO salesperson VALUES (1, 'John'), (2, 'Jane'); "
        "CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL); "
        "INSERT INTO timber_sales VALUES (1, 1, 120), (2, 2, 180);"
    ),
    "sql": (
        "SELECT s.name, SUM(t.volume) FROM timber_sales t "
        "JOIN salesperson s ON t.salesperson_id = s.salesperson_id GROUP BY s.name"
    ),
    "domain": "forestry",
}


def test_valid_row_is_accepted():
    example, reason = process_row(GOOD_ROW)
    assert example is not None, reason
    assert example.difficulty == "medium"
    assert "salesperson" in example.tables
    assert "TABLE salesperson" in example.schema_text
    assert example.setup_sql == GOOD_ROW["sql_context"]


def test_data_manipulation_task_type_excluded():
    row = dict(GOOD_ROW, sql_task_type="data manipulation")
    example, reason = process_row(row)
    assert example is None
    assert reason.startswith("task_type_excluded")


def test_data_definition_task_type_excluded():
    row = dict(GOOD_ROW, sql_task_type="data definition")
    example, reason = process_row(row)
    assert example is None
    assert reason.startswith("task_type_excluded")


def test_missing_fields_rejected():
    row = dict(GOOD_ROW, sql_prompt="")
    example, reason = process_row(row)
    assert example is None
    assert reason == "missing_field"


def test_broken_schema_setup_rejected():
    row = dict(GOOD_ROW, sql_context="THIS IS NOT VALID SQL AT ALL (((")
    example, reason = process_row(row)
    assert example is None
    assert reason == "schema_setup_failed"


def test_dangerous_target_sql_rejected():
    row = dict(GOOD_ROW, sql="DROP TABLE salesperson")
    example, reason = process_row(row)
    assert example is None
    assert reason.startswith("safety")


def test_sql_referencing_nonexistent_table_rejected():
    row = dict(GOOD_ROW, sql="SELECT * FROM made_up_table_xyz")
    example, reason = process_row(row)
    assert example is None
    assert reason.startswith("safety")


def test_difficulty_mapping_basic_sql_is_easy():
    row = dict(GOOD_ROW, sql_complexity="basic SQL", sql="SELECT * FROM salesperson")
    example, reason = process_row(row)
    assert example is not None, reason
    assert example.difficulty == "easy"


def test_difficulty_mapping_subqueries_is_hard():
    row = dict(
        GOOD_ROW,
        sql_complexity="subqueries",
        sql="SELECT * FROM salesperson WHERE salesperson_id IN (SELECT salesperson_id FROM timber_sales)",
    )
    example, reason = process_row(row)
    assert example is not None, reason
    assert example.difficulty == "hard"
