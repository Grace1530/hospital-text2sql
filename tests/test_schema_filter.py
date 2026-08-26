import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema_filter import select_relevant_tables

SCHEMA = {
    "tables": {
        "doctors": {"columns": [{"name": "doctor_id"}, {"name": "department_id"}, {"name": "first_name"}]},
        "departments": {"columns": [{"name": "department_id"}, {"name": "department_name"}]},
        "patients": {"columns": [{"name": "patient_id"}, {"name": "first_name"}, {"name": "address"}]},
    },
    "relationships": [
        {"from_table": "doctors", "from_column": "department_id", "to_table": "departments", "to_column": "department_id"},
    ],
}


def test_selects_directly_mentioned_table():
    result = select_relevant_tables("How many doctors are there?", SCHEMA)
    assert "doctors" in result


def test_expands_to_fk_referenced_table():
    result = select_relevant_tables("How many doctors work in Cardiology department?", SCHEMA)
    assert "doctors" in result
    assert "departments" in result  # pulled in via FK expansion


def test_falls_back_to_all_tables_when_nothing_matches():
    result = select_relevant_tables("asdf qwer zxcv", SCHEMA)
    assert set(result) == set(SCHEMA["tables"].keys())


def test_max_tables_caps_output():
    result = select_relevant_tables("doctors patients departments", SCHEMA, max_tables=1)
    assert len(result) == 1
