"""
Declarative mapping from the raw Kaggle T-SQL schema to the clean,
consistent snake_case DuckDB schema used by the rest of this project.

Design decisions (documented, not silently applied):

- Every raw table becomes exactly one clean table (no merges/splits), so the
  application/model always maps 1:1 back to a real, inspectable table.
- All identifiers are normalized to snake_case (dept_Id -> department_id,
  FName -> first_name, contact_No / conatct_No (typo) -> contact_no, ...).
- BedRecords.admission_Id and RoomRecords.admisson_ID (typo in raw) both
  become `admission_id` for consistency, but the two tables are KEPT
  SEPARATE (they represent different admission types / ID spaces in the
  source data — merging them would be a schema redesign beyond cleanup).
- Doctor.office_No -> doctors.office_no (nullable FK -> rooms.room_no):
  not every doctor has an assigned office.
- No canonical-to-raw translation layer is introduced anywhere else in the
  project: the Transformer generates SQL directly against these clean
  table/column names, and this module is only used once, at database build
  time, to go from the raw dump to the clean DuckDB file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ColumnMap:
    raw_name: str
    clean_name: str
    duckdb_type: str


@dataclass
class TableMap:
    raw_name: str
    clean_name: str
    columns: list[ColumnMap]
    primary_key: str
    foreign_keys: list[tuple[str, str, str]]  # (clean_column, ref_clean_table, ref_clean_column)


TABLE_MAPS: list[TableMap] = [
    TableMap(
        raw_name="Department",
        clean_name="departments",
        columns=[
            ColumnMap("dept_Id", "department_id", "INTEGER"),
            ColumnMap("dept_Name", "department_name", "VARCHAR"),
        ],
        primary_key="department_id",
        foreign_keys=[],
    ),
    TableMap(
        raw_name="Room",
        clean_name="rooms",
        columns=[
            ColumnMap("room_No", "room_no", "INTEGER"),
            ColumnMap("dept_Id", "department_id", "INTEGER"),
            ColumnMap("room_Type", "room_type", "VARCHAR"),
        ],
        primary_key="room_no",
        foreign_keys=[("department_id", "departments", "department_id")],
    ),
    TableMap(
        raw_name="Ward",
        clean_name="wards",
        columns=[
            ColumnMap("ward_No", "ward_no", "INTEGER"),
            ColumnMap("ward_Name", "ward_name", "VARCHAR"),
            ColumnMap("dept_Id", "department_id", "INTEGER"),
        ],
        primary_key="ward_no",
        foreign_keys=[("department_id", "departments", "department_id")],
    ),
    TableMap(
        raw_name="Bed",
        clean_name="beds",
        columns=[
            ColumnMap("bed_No", "bed_no", "INTEGER"),
            ColumnMap("ward_No", "ward_no", "INTEGER"),
        ],
        primary_key="bed_no",
        foreign_keys=[("ward_no", "wards", "ward_no")],
    ),
    TableMap(
        raw_name="Doctor",
        clean_name="doctors",
        columns=[
            ColumnMap("doct_Id", "doctor_id", "INTEGER"),
            ColumnMap("dept_Id", "department_id", "INTEGER"),
            ColumnMap("FName", "first_name", "VARCHAR"),
            ColumnMap("LName", "last_name", "VARCHAR"),
            ColumnMap("Gender", "gender", "VARCHAR(1)"),
            ColumnMap("contact_No", "contact_no", "VARCHAR"),
            ColumnMap("surgeon_Type", "surgeon_type", "VARCHAR"),
            ColumnMap("office_No", "office_no", "INTEGER"),
        ],
        primary_key="doctor_id",
        foreign_keys=[
            ("department_id", "departments", "department_id"),
            ("office_no", "rooms", "room_no"),
        ],
    ),
    TableMap(
        raw_name="Nurse",
        clean_name="nurses",
        columns=[
            ColumnMap("nurse_Id", "nurse_id", "INTEGER"),
            ColumnMap("dept_Id", "department_id", "INTEGER"),
            ColumnMap("FName", "first_name", "VARCHAR"),
            ColumnMap("LName", "last_name", "VARCHAR"),
            ColumnMap("Gender", "gender", "VARCHAR(1)"),
            ColumnMap("conatct_No", "contact_no", "VARCHAR"),
        ],
        primary_key="nurse_id",
        foreign_keys=[("department_id", "departments", "department_id")],
    ),
    TableMap(
        raw_name="Helpers",
        clean_name="helpers",
        columns=[
            ColumnMap("helper_Id", "helper_id", "INTEGER"),
            ColumnMap("dept_Id", "department_id", "INTEGER"),
            ColumnMap("FName", "first_name", "VARCHAR"),
            ColumnMap("LName", "last_name", "VARCHAR"),
            ColumnMap("Gender", "gender", "VARCHAR(1)"),
            ColumnMap("contact_No", "contact_no", "VARCHAR"),
        ],
        primary_key="helper_id",
        foreign_keys=[("department_id", "departments", "department_id")],
    ),
    TableMap(
        raw_name="Patients",
        clean_name="patients",
        columns=[
            ColumnMap("patient_Id", "patient_id", "INTEGER"),
            ColumnMap("FName", "first_name", "VARCHAR"),
            ColumnMap("LName", "last_name", "VARCHAR"),
            ColumnMap("Gender", "gender", "VARCHAR(1)"),
            ColumnMap("Date_Of_Birth", "date_of_birth", "DATE"),
            ColumnMap("contact_No", "contact_no", "VARCHAR"),
            ColumnMap("pt_Address", "address", "VARCHAR"),
        ],
        primary_key="patient_id",
        foreign_keys=[],
    ),
    TableMap(
        raw_name="BedRecords",
        clean_name="bed_records",
        columns=[
            ColumnMap("admission_Id", "admission_id", "INTEGER"),
            ColumnMap("bed_No", "bed_no", "INTEGER"),
            ColumnMap("patient_Id", "patient_id", "INTEGER"),
            ColumnMap("nurse_Id", "nurse_id", "INTEGER"),
            ColumnMap("helper_Id", "helper_id", "INTEGER"),
            ColumnMap("admission_Date", "admission_date", "DATE"),
            ColumnMap("discharge_Date", "discharge_date", "DATE"),
            ColumnMap("amount", "amount", "INTEGER"),
            ColumnMap("mode_of_payment", "mode_of_payment", "VARCHAR"),
        ],
        primary_key="admission_id",
        foreign_keys=[
            ("bed_no", "beds", "bed_no"),
            ("patient_id", "patients", "patient_id"),
            ("nurse_id", "nurses", "nurse_id"),
            ("helper_id", "helpers", "helper_id"),
        ],
    ),
    TableMap(
        raw_name="RoomRecords",
        clean_name="room_records",
        columns=[
            ColumnMap("admisson_ID", "admission_id", "INTEGER"),
            ColumnMap("room_no", "room_no", "INTEGER"),
            ColumnMap("patient_Id", "patient_id", "INTEGER"),
            ColumnMap("nurse_Id", "nurse_id", "INTEGER"),
            ColumnMap("helper_Id", "helper_id", "INTEGER"),
            ColumnMap("admission_Date", "admission_date", "DATE"),
            ColumnMap("discharge_Date", "discharge_date", "DATE"),
            ColumnMap("amount", "amount", "INTEGER"),
            ColumnMap("mode_of_payment", "mode_of_payment", "VARCHAR"),
        ],
        primary_key="admission_id",
        foreign_keys=[
            ("room_no", "rooms", "room_no"),
            ("patient_id", "patients", "patient_id"),
            ("nurse_id", "nurses", "nurse_id"),
            ("helper_id", "helpers", "helper_id"),
        ],
    ),
    TableMap(
        raw_name="Appointment",
        clean_name="appointments",
        columns=[
            ColumnMap("appoIntment_Id", "appointment_id", "INTEGER"),
            ColumnMap("patient_Id", "patient_id", "INTEGER"),
            ColumnMap("doct_Id", "doctor_id", "INTEGER"),
            ColumnMap("reason", "reason", "VARCHAR"),
            ColumnMap("appointment_Date", "appointment_date", "DATE"),
            ColumnMap("payment_amount", "payment_amount", "INTEGER"),
            ColumnMap("mode_of_payment", "mode_of_payment", "VARCHAR"),
            ColumnMap("mode_of_appointment", "mode_of_appointment", "VARCHAR"),
            ColumnMap("appointment_status", "appointment_status", "VARCHAR"),
        ],
        primary_key="appointment_id",
        foreign_keys=[
            ("patient_id", "patients", "patient_id"),
            ("doctor_id", "doctors", "doctor_id"),
        ],
    ),
    TableMap(
        raw_name="MedicalRecord",
        clean_name="medical_records",
        columns=[
            ColumnMap("record_Id", "record_id", "INTEGER"),
            ColumnMap("doct_Id", "doctor_id", "INTEGER"),
            ColumnMap("patient_Id", "patient_id", "INTEGER"),
            ColumnMap("visit_Date", "visit_date", "DATE"),
            ColumnMap("curr_Weight", "weight_kg", "DECIMAL(10,2)"),
            ColumnMap("curr_height", "height_cm", "DECIMAL(10,2)"),
            ColumnMap("curr_Blood_Pressure", "blood_pressure", "VARCHAR"),
            ColumnMap("curr_Temp_F", "temperature_f", "DECIMAL(10,2)"),
            ColumnMap("diagnosis", "diagnosis", "VARCHAR"),
            ColumnMap("treatment", "treatment", "VARCHAR"),
            ColumnMap("next_Visit", "next_visit_date", "DATE"),
        ],
        primary_key="record_id",
        foreign_keys=[
            ("doctor_id", "doctors", "doctor_id"),
            ("patient_id", "patients", "patient_id"),
        ],
    ),
    TableMap(
        raw_name="StaffShift",
        clean_name="staff_shifts",
        columns=[
            ColumnMap("shift_Id", "shift_id", "INTEGER"),
            ColumnMap("doct_Id", "doctor_id", "INTEGER"),
            ColumnMap("nurse_Id", "nurse_id", "INTEGER"),
            ColumnMap("helper_Id", "helper_id", "INTEGER"),
            ColumnMap("shift_Date", "shift_date", "DATE"),
            ColumnMap("shift_Start", "shift_start", "TIME"),
            ColumnMap("shift_End", "shift_end", "TIME"),
        ],
        primary_key="shift_id",
        foreign_keys=[
            ("doctor_id", "doctors", "doctor_id"),
            ("nurse_id", "nurses", "nurse_id"),
            ("helper_id", "helpers", "helper_id"),
        ],
    ),
    TableMap(
        raw_name="SurgeryRecord",
        clean_name="surgery_records",
        columns=[
            ColumnMap("surgery_Id", "surgery_id", "INTEGER"),
            ColumnMap("patient_Id", "patient_id", "INTEGER"),
            ColumnMap("surgeon_Id", "surgeon_id", "INTEGER"),
            ColumnMap("surgery_Type", "surgery_type", "VARCHAR"),
            ColumnMap("surgery_Date", "surgery_date", "DATE"),
            ColumnMap("start_Time", "start_time", "TIME"),
            ColumnMap("end_Time", "end_time", "TIME"),
            ColumnMap("room_no", "room_no", "INTEGER"),
            ColumnMap("notes", "notes", "VARCHAR"),
            ColumnMap("nurse_Id", "nurse_id", "INTEGER"),
            ColumnMap("helper_Id", "helper_id", "INTEGER"),
        ],
        primary_key="surgery_id",
        foreign_keys=[
            ("patient_id", "patients", "patient_id"),
            ("surgeon_id", "doctors", "doctor_id"),
            ("room_no", "rooms", "room_no"),
            ("nurse_id", "nurses", "nurse_id"),
            ("helper_id", "helpers", "helper_id"),
        ],
    ),
]

# Build order respecting foreign key dependencies (parents before children).
BUILD_ORDER = [
    "departments",
    "rooms",
    "wards",
    "beds",
    "doctors",
    "nurses",
    "helpers",
    "patients",
    "bed_records",
    "room_records",
    "appointments",
    "medical_records",
    "staff_shifts",
    "surgery_records",
]

TABLE_MAPS_BY_CLEAN_NAME = {t.clean_name: t for t in TABLE_MAPS}

assert set(BUILD_ORDER) == set(TABLE_MAPS_BY_CLEAN_NAME.keys())
