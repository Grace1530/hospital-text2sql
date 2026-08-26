# Raw Dataset Inspection Report

Source: `data/raw/Hospital_Management_System.sql` (Kaggle hospital-management dataset, untouched).

Dialect: **Microsoft SQL Server (T-SQL)** — evidence: `CREATE DATABASE`, `GO` batch separators, `Time` column type, no backtick identifiers.

## Data quality findings

1. **Missing statement terminator.** The `Doctor` INSERT block has no trailing `;` before `Insert Into Nurse`. T-SQL does not require semicolons (statement boundaries are keyword-based), so this is valid T-SQL but breaks naive semicolon-based SQL splitters. Our parser (`src/data/raw_sql_parser.py`) splits on statement-start keywords instead, which handles this correctly.
2. **Intentionally-disabled rows.** Some INSERT blocks (e.g. `Room`) contain rows wrapped in `/* ... */` block comments — the dataset author disabled some generated rows. These are correctly excluded by comment stripping.
3. **Typo'd column names** carried over from the source: `Nurse.conatct_No` (should be `contact_No`), `RoomRecords.admisson_ID` (should be `admission_ID`), `Appointment.appoIntment_Id` (inconsistent casing). These are normalized in the clean schema.
4. **Inconsistent PK naming across near-duplicate tables**: `BedRecords.admission_Id` vs `RoomRecords.admisson_ID` — same concept, different raw names. Normalized to `admission_id` in both (still separate tables/ID spaces, not merged).

## Tables

### `Department` (31 rows)

| column | type | notes |
|---|---|---|
| dept_Id | Int | PRIMARY KEY |
| dept_Name | Varchar(100) |  |

### `Room` (391 rows)

| column | type | notes |
|---|---|---|
| room_No | Int | PRIMARY KEY |
| dept_Id | Int | FK -> Department.dept_Id |
| room_Type | Varchar(100) |  |

### `Doctor` (400 rows)

| column | type | notes |
|---|---|---|
| doct_Id | Int | PRIMARY KEY |
| dept_Id | Int | FK -> Department.dept_Id |
| FName | Varchar(100) |  |
| LName | Varchar(100) |  |
| Gender | CHAR |  |
| contact_No | Varchar(100) |  |
| surgeon_Type | Varchar(100) | 362 NULLs |
| office_No | Int | FK -> Room.room_No, 38 NULLs |

### `Nurse` (500 rows)

| column | type | notes |
|---|---|---|
| nurse_Id | Int | PRIMARY KEY |
| dept_Id | Int | FK -> Department.dept_Id |
| FName | Varchar(100) |  |
| LName | Varchar(100) |  |
| Gender | Char |  |
| conatct_No | Varchar(100) |  |

### `Helpers` (1100 rows)

| column | type | notes |
|---|---|---|
| helper_Id | Int | PRIMARY KEY |
| dept_Id | Int | FK -> Department.dept_Id |
| FName | Varchar(100) |  |
| LName | Varchar(100) |  |
| Gender | Char |  |
| contact_No | Varchar(100) |  |

### `Ward` (63 rows)

| column | type | notes |
|---|---|---|
| ward_No | Int | PRIMARY KEY |
| ward_Name | Varchar(100) |  |
| dept_Id | Int | FK -> Department.dept_Id |

### `Bed` (500 rows)

| column | type | notes |
|---|---|---|
| bed_No | Int | PRIMARY KEY |
| ward_No | Int | FK -> Ward.ward_No |

### `Patients` (1500 rows)

| column | type | notes |
|---|---|---|
| patient_Id | Int | PRIMARY KEY |
| FName | Varchar(100) |  |
| LName | Varchar(100) |  |
| Gender | Char |  |
| Date_Of_Birth | Date |  |
| contact_No | Varchar(100) |  |
| pt_Address | Varchar(100) |  |

### `BedRecords` (1000 rows)

| column | type | notes |
|---|---|---|
| admission_Id | Int | PRIMARY KEY |
| bed_No | Int | FK -> Bed.bed_No |
| patient_Id | Int | FK -> Patients.patient_Id |
| nurse_Id | Int | FK -> Nurse.nurse_Id |
| helper_Id | Int | FK -> Helpers.helper_Id |
| admission_Date | Date |  |
| discharge_Date | Date |  |
| amount | Int |  |
| mode_of_payment | Varchar(50) |  |

### `RoomRecords` (1000 rows)

| column | type | notes |
|---|---|---|
| admisson_ID | Int | PRIMARY KEY |
| room_no | Int | FK -> Room.room_No |
| patient_Id | Int | FK -> Patients.patient_Id |
| nurse_Id | Int | FK -> Nurse.nurse_Id |
| helper_Id | Int | FK -> Helpers.helper_Id |
| admission_Date | Date |  |
| discharge_Date | Date |  |
| amount | Int |  |
| mode_of_payment | Varchar(50) |  |

### `Appointment` (1000 rows)

| column | type | notes |
|---|---|---|
| appoIntment_Id | Int | PRIMARY KEY |
| patient_Id | Int | FK -> Patients.patient_Id |
| doct_Id | Int | FK -> Doctor.doct_Id |
| reason | Varchar(100) |  |
| appointment_Date | Date |  |
| payment_amount | Int |  |
| mode_of_payment | Varchar(100) |  |
| mode_of_appointment | Varchar(100) |  |
| appointment_status | Varchar(100) |  |

### `MedicalRecord` (3000 rows)

| column | type | notes |
|---|---|---|
| record_Id | Int | PRIMARY KEY |
| doct_Id | Int | FK -> Doctor.doct_Id |
| patient_Id | Int | FK -> Patients.patient_Id |
| visit_Date | Date |  |
| curr_Weight | Decimal(10,2) |  |
| curr_height | Decimal(10,2) |  |
| curr_Blood_Pressure | Varchar(100) |  |
| curr_Temp_F | Decimal(10,2) |  |
| diagnosis | Varchar(500) |  |
| treatment | Varchar(100) |  |
| next_Visit | Date | 1531 NULLs |

### `StaffShift` (2058 rows)

| column | type | notes |
|---|---|---|
| shift_Id | Int | PRIMARY KEY |
| doct_Id | Int | FK -> Doctor.doct_Id, 1000 NULLs |
| nurse_Id | Int | FK -> Nurse.nurse_Id, 1558 NULLs |
| helper_Id | Int | FK -> Helpers.helper_Id, 1558 NULLs |
| shift_Date | Date |  |
| shift_Start | Time |  |
| shift_End | Time |  |

### `SurgeryRecord` (1000 rows)

| column | type | notes |
|---|---|---|
| surgery_Id | Int | PRIMARY KEY |
| patient_Id | Int | FK -> Patients.patient_Id |
| surgeon_Id | Int | FK -> Doctor.doct_Id |
| surgery_Type | Varchar(100) |  |
| surgery_Date | Date |  |
| start_Time | Time |  |
| end_Time | Time |  |
| room_no | Int | FK -> Room.room_No |
| notes | Varchar(1000) |  |
| nurse_Id | Int | FK -> Nurse.nurse_Id |
| helper_Id | Int | FK -> Helpers.helper_Id |

## Referential integrity

| table.column | -> ref_table.ref_column | dangling (non-null, no match) |
|---|---|---|
| Room.dept_Id | Department.dept_Id | 0 |
| Doctor.dept_Id | Department.dept_Id | 0 |
| Doctor.office_No | Room.room_No | 0 |
| Nurse.dept_Id | Department.dept_Id | 0 |
| Helpers.dept_Id | Department.dept_Id | 0 |
| Ward.dept_Id | Department.dept_Id | 0 |
| Bed.ward_No | Ward.ward_No | 0 |
| BedRecords.bed_No | Bed.bed_No | 0 |
| BedRecords.patient_Id | Patients.patient_Id | 0 |
| BedRecords.nurse_Id | Nurse.nurse_Id | 0 |
| BedRecords.helper_Id | Helpers.helper_Id | 0 |
| RoomRecords.room_no | Room.room_No | 0 |
| RoomRecords.patient_Id | Patients.patient_Id | 0 |
| RoomRecords.nurse_Id | Nurse.nurse_Id | 0 |
| RoomRecords.helper_Id | Helpers.helper_Id | 0 |
| Appointment.patient_Id | Patients.patient_Id | 0 |
| Appointment.doct_Id | Doctor.doct_Id | 0 |
| MedicalRecord.doct_Id | Doctor.doct_Id | 0 |
| MedicalRecord.patient_Id | Patients.patient_Id | 0 |
| StaffShift.doct_Id | Doctor.doct_Id | 0 |
| StaffShift.nurse_Id | Nurse.nurse_Id | 0 |
| StaffShift.helper_Id | Helpers.helper_Id | 0 |
| SurgeryRecord.patient_Id | Patients.patient_Id | 0 |
| SurgeryRecord.surgeon_Id | Doctor.doct_Id | 0 |
| SurgeryRecord.room_no | Room.room_No | 0 |
| SurgeryRecord.nurse_Id | Nurse.nurse_Id | 0 |
| SurgeryRecord.helper_Id | Helpers.helper_Id | 0 |