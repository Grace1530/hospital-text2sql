import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sql.safety import validate_sql_safety

KNOWN_TABLES = {
    "departments", "rooms", "wards", "beds", "doctors", "nurses", "helpers",
    "patients", "bed_records", "room_records", "appointments",
    "medical_records", "staff_shifts", "surgery_records",
}


def test_valid_select_passes():
    r = validate_sql_safety(
        "SELECT COUNT(*) FROM doctors d JOIN departments dep "
        "ON d.department_id = dep.department_id WHERE dep.department_name = 'Cardiology'",
        known_tables=KNOWN_TABLES,
    )
    assert r.is_safe


def test_cte_with_known_base_table_passes():
    r = validate_sql_safety(
        "WITH t AS (SELECT * FROM patients) SELECT * FROM t",
        known_tables=KNOWN_TABLES,
    )
    assert r.is_safe, r.reason


def test_union_of_selects_passes():
    r = validate_sql_safety(
        "SELECT patient_id FROM patients UNION SELECT patient_id FROM patients",
        known_tables=KNOWN_TABLES,
    )
    assert r.is_safe


def test_drop_table_rejected():
    r = validate_sql_safety("DROP TABLE doctors", known_tables=KNOWN_TABLES)
    assert not r.is_safe


def test_insert_rejected():
    r = validate_sql_safety("INSERT INTO doctors VALUES (1)", known_tables=KNOWN_TABLES)
    assert not r.is_safe


def test_update_rejected():
    r = validate_sql_safety("UPDATE doctors SET first_name = 'x'", known_tables=KNOWN_TABLES)
    assert not r.is_safe


def test_delete_rejected():
    r = validate_sql_safety("DELETE FROM doctors", known_tables=KNOWN_TABLES)
    assert not r.is_safe


def test_stacked_statements_rejected():
    r = validate_sql_safety("SELECT * FROM doctors; DROP TABLE doctors;", known_tables=KNOWN_TABLES)
    assert not r.is_safe


def test_attach_rejected():
    r = validate_sql_safety("ATTACH 'evil.db' AS evil", known_tables=KNOWN_TABLES)
    assert not r.is_safe


def test_pragma_rejected():
    r = validate_sql_safety("PRAGMA database_list", known_tables=KNOWN_TABLES)
    assert not r.is_safe


def test_unknown_table_rejected():
    r = validate_sql_safety("SELECT * FROM secret_admin_table", known_tables=KNOWN_TABLES)
    assert not r.is_safe


def test_empty_sql_rejected():
    r = validate_sql_safety("", known_tables=KNOWN_TABLES)
    assert not r.is_safe


def test_syntax_error_rejected():
    r = validate_sql_safety("SELECT FROM WHERE", known_tables=KNOWN_TABLES)
    assert not r.is_safe
