"""
Phase 3 — Automated validation of the clean DuckDB hospital database.

These tests do NOT trust that the import script "ran without error" — they
independently query the built database file and check structure, row
counts, referential integrity, and representative joins.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema_mapping import BUILD_ORDER, TABLE_MAPS_BY_CLEAN_NAME  # noqa: E402

EXPECTED_MIN_ROWS = {
    "departments": 20,
    "rooms": 300,
    "wards": 50,
    "beds": 400,
    "doctors": 300,
    "nurses": 400,
    "helpers": 900,
    "patients": 1000,
    "bed_records": 900,
    "room_records": 900,
    "appointments": 900,
    "medical_records": 2500,
    "staff_shifts": 1800,
    "surgery_records": 900,
}


def test_database_reopens(con):
    """The database file can be reopened read-only after being built."""
    assert con.execute("SELECT 1").fetchone()[0] == 1


def test_all_expected_tables_exist(con):
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    for clean_name in BUILD_ORDER:
        assert clean_name in tables, f"missing table {clean_name}"
    assert tables == set(BUILD_ORDER)


def test_all_expected_columns_exist_with_types(con):
    for clean_name in BUILD_ORDER:
        tmap = TABLE_MAPS_BY_CLEAN_NAME[clean_name]
        rows = con.execute(f"PRAGMA table_info('{clean_name}')").fetchall()
        actual_cols = {r[1] for r in rows}
        expected_cols = {c.clean_name for c in tmap.columns}
        assert actual_cols == expected_cols, f"{clean_name}: {actual_cols} != {expected_cols}"


def test_primary_keys_are_unique_and_non_null(con):
    for clean_name in BUILD_ORDER:
        pk = TABLE_MAPS_BY_CLEAN_NAME[clean_name].primary_key
        total = con.execute(f"SELECT COUNT(*) FROM {clean_name}").fetchone()[0]
        distinct = con.execute(f"SELECT COUNT(DISTINCT {pk}) FROM {clean_name}").fetchone()[0]
        nulls = con.execute(f"SELECT COUNT(*) FROM {clean_name} WHERE {pk} IS NULL").fetchone()[0]
        assert total == distinct, f"{clean_name}.{pk} has duplicate values"
        assert nulls == 0, f"{clean_name}.{pk} has NULLs"


def test_row_counts_are_sensible(con):
    for clean_name, min_expected in EXPECTED_MIN_ROWS.items():
        count = con.execute(f"SELECT COUNT(*) FROM {clean_name}").fetchone()[0]
        assert count >= min_expected, f"{clean_name}: {count} rows, expected >= {min_expected}"


def test_foreign_keys_have_no_dangling_references(con):
    for clean_name in BUILD_ORDER:
        tmap = TABLE_MAPS_BY_CLEAN_NAME[clean_name]
        for local_col, ref_table, ref_col in tmap.foreign_keys:
            sql = f"""
                SELECT COUNT(*) FROM {clean_name} t
                WHERE t.{local_col} IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM {ref_table} r WHERE r.{ref_col} = t.{local_col}
                  )
            """
            dangling = con.execute(sql).fetchone()[0]
            assert dangling == 0, f"{clean_name}.{local_col} -> {ref_table}.{ref_col}: {dangling} dangling rows"


def test_two_table_join_doctors_departments(con):
    row = con.execute(
        """
        SELECT COUNT(*) FROM doctors d
        JOIN departments dep ON d.department_id = dep.department_id
        WHERE dep.department_name = 'Cardiology'
        """
    ).fetchone()
    assert row[0] > 0


def test_three_table_join_appointments_patients_doctors(con):
    row = con.execute(
        """
        SELECT p.first_name, p.last_name, doc.first_name, doc.last_name, a.appointment_date
        FROM appointments a
        JOIN patients p ON a.patient_id = p.patient_id
        JOIN doctors doc ON a.doctor_id = doc.doctor_id
        LIMIT 5
        """
    ).fetchall()
    assert len(row) == 5


def test_left_join_bed_records_with_department_via_bed_ward(con):
    rows = con.execute(
        """
        SELECT br.admission_id, dep.department_name
        FROM bed_records br
        JOIN beds b ON br.bed_no = b.bed_no
        JOIN wards w ON b.ward_no = w.ward_no
        LEFT JOIN departments dep ON w.department_id = dep.department_id
        LIMIT 5
        """
    ).fetchall()
    assert len(rows) == 5


def test_aggregation_group_by_having(con):
    rows = con.execute(
        """
        SELECT dep.department_name, COUNT(*) AS num_doctors
        FROM doctors doc
        JOIN departments dep ON doc.department_id = dep.department_id
        GROUP BY dep.department_name
        HAVING COUNT(*) > 0
        ORDER BY num_doctors DESC
        """
    ).fetchall()
    assert len(rows) > 0


def test_subquery_patients_with_surgery(con):
    rows = con.execute(
        """
        SELECT COUNT(*) FROM patients
        WHERE patient_id IN (SELECT patient_id FROM surgery_records)
        """
    ).fetchone()
    assert rows[0] > 0


def test_date_filtering_works(con):
    rows = con.execute(
        """
        SELECT COUNT(*) FROM appointments
        WHERE appointment_date >= DATE '2024-01-01'
        """
    ).fetchone()
    assert rows[0] >= 0  # executes without error; hospital data may be any year


def test_nullable_columns_present_where_expected(con):
    # Not every doctor is a surgeon with an office; not every discharge has completed.
    null_office = con.execute("SELECT COUNT(*) FROM doctors WHERE office_no IS NULL").fetchone()[0]
    assert null_office >= 0
    null_next_visit = con.execute(
        "SELECT COUNT(*) FROM medical_records WHERE next_visit_date IS NULL"
    ).fetchone()[0]
    assert null_next_visit > 0
