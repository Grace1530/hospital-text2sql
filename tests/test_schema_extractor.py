import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema_extractor import extract_schema, serialize_schema_text

DB_PATH = Path(__file__).resolve().parents[1] / "database" / "hospital.duckdb"


@pytest.fixture(scope="module")
def schema():
    if not DB_PATH.exists():
        pytest.skip("database not built")
    return extract_schema(DB_PATH)


def test_extracts_all_fourteen_tables(schema):
    assert len(schema["tables"]) == 14


def test_doctors_table_has_expected_columns(schema):
    cols = {c["name"] for c in schema["tables"]["doctors"]["columns"]}
    assert cols == {
        "doctor_id", "department_id", "first_name", "last_name",
        "gender", "contact_no", "surgeon_type", "office_no",
    }


def test_primary_keys_detected(schema):
    doctor_pk = [c["name"] for c in schema["tables"]["doctors"]["columns"] if c["is_primary_key"]]
    assert doctor_pk == ["doctor_id"]


def test_foreign_keys_detected_in_relationships(schema):
    rels = schema["relationships"]
    assert any(r["from_table"] == "doctors" and r["to_table"] == "departments" for r in rels)
    assert any(r["from_table"] == "doctors" and r["to_table"] == "rooms" for r in rels)


def test_row_counts_present_and_positive(schema):
    for tname, tinfo in schema["tables"].items():
        assert tinfo["row_count"] > 0, f"{tname} has zero rows"


def test_serialize_schema_text_full():
    fake_schema = {
        "tables": {
            "doctors": {"columns": [{"name": "doctor_id"}, {"name": "first_name"}]},
            "departments": {"columns": [{"name": "department_id"}]},
        }
    }
    text = serialize_schema_text(fake_schema)
    assert "TABLE doctors" in text
    assert "- doctor_id" in text
    assert "TABLE departments" in text


def test_serialize_schema_text_filtered_subset():
    fake_schema = {
        "tables": {
            "doctors": {"columns": [{"name": "doctor_id"}]},
            "departments": {"columns": [{"name": "department_id"}]},
        }
    }
    text = serialize_schema_text(fake_schema, table_names=["doctors"])
    assert "TABLE doctors" in text
    assert "TABLE departments" not in text
