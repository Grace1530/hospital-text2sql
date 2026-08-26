# Database Build Report

Built from `data/raw/Hospital_Management_System.sql` into `database/hospital.duckdb`.

| clean table | raw table | rows loaded | primary key | foreign keys |
|---|---|---|---|---|
| departments | Department | 31 | department_id | - |
| rooms | Room | 391 | room_no | department_id->departments.department_id |
| wards | Ward | 63 | ward_no | department_id->departments.department_id |
| beds | Bed | 500 | bed_no | ward_no->wards.ward_no |
| doctors | Doctor | 400 | doctor_id | department_id->departments.department_id; office_no->rooms.room_no |
| nurses | Nurse | 500 | nurse_id | department_id->departments.department_id |
| helpers | Helpers | 1100 | helper_id | department_id->departments.department_id |
| patients | Patients | 1500 | patient_id | - |
| bed_records | BedRecords | 1000 | admission_id | bed_no->beds.bed_no; patient_id->patients.patient_id; nurse_id->nurses.nurse_id; helper_id->helpers.helper_id |
| room_records | RoomRecords | 1000 | admission_id | room_no->rooms.room_no; patient_id->patients.patient_id; nurse_id->nurses.nurse_id; helper_id->helpers.helper_id |
| appointments | Appointment | 1000 | appointment_id | patient_id->patients.patient_id; doctor_id->doctors.doctor_id |
| medical_records | MedicalRecord | 3000 | record_id | doctor_id->doctors.doctor_id; patient_id->patients.patient_id |
| staff_shifts | StaffShift | 2058 | shift_id | doctor_id->doctors.doctor_id; nurse_id->nurses.nurse_id; helper_id->helpers.helper_id |
| surgery_records | SurgeryRecord | 1000 | surgery_id | patient_id->patients.patient_id; surgeon_id->doctors.doctor_id; room_no->rooms.room_no; nurse_id->nurses.nurse_id; helper_id->helpers.helper_id |

**Total rows loaded: 13543**