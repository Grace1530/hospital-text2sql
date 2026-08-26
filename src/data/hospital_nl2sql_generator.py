"""
Phase 5 — Hospital-specific Text-to-SQL example generation.

Every example is generated from a (question-template, SQL-template) pair,
parametrized with REAL values sampled from the live clean database
(department names, doctor names, dates, ...), so the generated SQL is
guaranteed to reference values that actually exist. Every single generated
example is then EXECUTED against database/hospital.duckdb; only examples
that execute successfully are kept (see verify.py).

Templates cover (per the project spec): SELECT, WHERE, comparison
operators, AND, OR, LIKE, BETWEEN, COUNT, SUM, AVG, MIN, MAX, GROUP BY,
HAVING, ORDER BY, LIMIT, DISTINCT, 2-table JOIN, 3-table JOIN, multi-JOIN,
LEFT JOIN, subqueries, date filtering — across patients, doctors,
departments, nurses, helpers, appointments, medical records, surgeries,
wards, rooms, beds, admissions, and staff shifts.

Each template supplies several natural-language PARAPHRASES so the corpus
doesn't collapse into "Show me..." repeated with different nouns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import duckdb


@dataclass
class Example:
    question: str
    sql: str
    category: str
    difficulty: str  # easy | medium | hard
    tables: list[str] = field(default_factory=list)


def _fetch_col(con, sql: str) -> list:
    return [r[0] for r in con.execute(sql).fetchall()]


def _sample(rng: random.Random, values: list, k: int) -> list:
    if not values:
        return []
    k = min(k, len(values))
    return rng.sample(values, k)


def _pick(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


# ---------------------------------------------------------------------------
# Reference value pools, pulled live from the database (not hand-invented).
# ---------------------------------------------------------------------------

@dataclass
class ValuePools:
    department_names: list[str]
    room_types: list[str]
    surgery_types: list[str]
    appointment_statuses: list[str]
    payment_modes: list[str]
    appointment_modes: list[str]
    ward_names: list[str]
    surgeon_types: list[str]
    genders: list[str]
    reasons: list[str]
    diagnoses: list[str]
    treatments: list[str]
    doctor_ids: list[int]
    patient_ids: list[int]
    doctor_last_names: list[str]
    patient_last_names: list[str]

    @classmethod
    def load(cls, con) -> "ValuePools":
        return cls(
            department_names=_fetch_col(con, "SELECT DISTINCT department_name FROM departments"),
            room_types=_fetch_col(con, "SELECT DISTINCT room_type FROM rooms"),
            surgery_types=_fetch_col(con, "SELECT DISTINCT surgery_type FROM surgery_records"),
            appointment_statuses=_fetch_col(con, "SELECT DISTINCT appointment_status FROM appointments"),
            payment_modes=_fetch_col(con, "SELECT DISTINCT mode_of_payment FROM appointments"),
            appointment_modes=_fetch_col(con, "SELECT DISTINCT mode_of_appointment FROM appointments"),
            ward_names=_fetch_col(con, "SELECT DISTINCT ward_name FROM wards"),
            surgeon_types=_fetch_col(con, "SELECT DISTINCT surgeon_type FROM doctors WHERE surgeon_type IS NOT NULL"),
            genders=_fetch_col(con, "SELECT DISTINCT gender FROM patients"),
            reasons=_fetch_col(con, "SELECT DISTINCT reason FROM appointments"),
            diagnoses=_fetch_col(con, "SELECT DISTINCT diagnosis FROM medical_records"),
            treatments=_fetch_col(con, "SELECT DISTINCT treatment FROM medical_records"),
            doctor_ids=_fetch_col(con, "SELECT doctor_id FROM doctors"),
            patient_ids=_fetch_col(con, "SELECT patient_id FROM patients"),
            doctor_last_names=_fetch_col(con, "SELECT DISTINCT last_name FROM doctors"),
            patient_last_names=_fetch_col(con, "SELECT DISTINCT last_name FROM patients"),
        )


# ---------------------------------------------------------------------------
# Template generator functions. Each returns a list of Example.
# `n` is the number of parameter samples to draw (best-effort; actual count
# may be lower if the value pool is small).
# ---------------------------------------------------------------------------

def gen_count_doctors_by_department(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many doctors work in {dept}?",
        "What is the number of doctors in the {dept} department?",
        "Count the doctors assigned to {dept}.",
        "Give me the total number of doctors in {dept}.",
    ]
    out = []
    for dept in _sample(rng, pools.department_names, n):
        q = _pick(rng, templates).format(dept=dept)
        sql = (
            "SELECT COUNT(*) FROM doctors d "
            "JOIN departments dep ON d.department_id = dep.department_id "
            f"WHERE dep.department_name = '{dept}'"
        )
        out.append(Example(q, sql, "aggregation_join", "medium", ["doctors", "departments"]))
    return out


def gen_list_doctors_by_department(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "List all doctors in the {dept} department.",
        "Who are the doctors working in {dept}?",
        "Show me the names of doctors in {dept}.",
    ]
    out = []
    for dept in _sample(rng, pools.department_names, n):
        q = _pick(rng, templates).format(dept=dept)
        sql = (
            "SELECT d.first_name, d.last_name FROM doctors d "
            "JOIN departments dep ON d.department_id = dep.department_id "
            f"WHERE dep.department_name = '{dept}'"
        )
        out.append(Example(q, sql, "join", "easy", ["doctors", "departments"]))
    return out


def gen_patients_by_gender(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many patients are {g}?",
        "Count the number of {g} patients.",
    ]
    out = []
    gender_word = {"M": "male", "F": "female"}
    for g in pools.genders:
        word = gender_word.get(g, g)
        q = _pick(rng, templates).format(g=word)
        sql = f"SELECT COUNT(*) FROM patients WHERE gender = '{g}'"
        out.append(Example(q, sql, "aggregation_filter", "easy", ["patients"]))
    return out


def gen_patients_born_between(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "Which patients were born between {y1} and {y2}?",
        "List patients with a date of birth between {y1}-01-01 and {y2}-12-31.",
        "Show patients born from {y1} to {y2}.",
    ]
    out = []
    for _ in range(n):
        y1 = rng.randint(1940, 2005)
        y2 = y1 + rng.randint(1, 15)
        q = _pick(rng, templates).format(y1=y1, y2=y2)
        sql = (
            "SELECT patient_id, first_name, last_name, date_of_birth FROM patients "
            f"WHERE date_of_birth BETWEEN '{y1}-01-01' AND '{y2}-12-31'"
        )
        out.append(Example(q, sql, "between_filter", "easy", ["patients"]))
    return out


def gen_patients_address_like(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    cities = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan", "Peshawar", "Rawalpindi", "Quetta"]
    templates = [
        "Which patients live in {city}?",
        "List patients whose address contains {city}.",
        "Show me patients from {city}.",
    ]
    out = []
    for city in _sample(rng, cities, n):
        q = _pick(rng, templates).format(city=city)
        sql = f"SELECT patient_id, first_name, last_name, address FROM patients WHERE address LIKE '%{city}%'"
        out.append(Example(q, sql, "like_filter", "easy", ["patients"]))
    return out


def gen_appointments_by_status_and_mode(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many appointments were {status} and paid by {mode}?",
        "Count appointments with status {status} using {mode} as payment.",
    ]
    out = []
    for _ in range(n):
        status = _pick(rng, pools.appointment_statuses)
        mode = _pick(rng, pools.payment_modes)
        q = _pick(rng, templates).format(status=status, mode=mode)
        sql = (
            "SELECT COUNT(*) FROM appointments "
            f"WHERE appointment_status = '{status}' AND mode_of_payment = '{mode}'"
        )
        out.append(Example(q, sql, "and_filter", "easy", ["appointments"]))
    return out


def gen_appointments_status_or_status(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many appointments are either {s1} or {s2}?",
        "Count appointments that are {s1} or {s2}.",
    ]
    out = []
    statuses = pools.appointment_statuses
    for _ in range(n):
        if len(statuses) < 2:
            break
        s1, s2 = rng.sample(statuses, 2)
        q = _pick(rng, templates).format(s1=s1, s2=s2)
        sql = f"SELECT COUNT(*) FROM appointments WHERE appointment_status = '{s1}' OR appointment_status = '{s2}'"
        out.append(Example(q, sql, "or_filter", "easy", ["appointments"]))
    return out


def gen_appointments_reason_count_group_by(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many appointments are there for each reason?",
        "Group appointments by reason and count them.",
        "Show the number of appointments per reason.",
    ]
    q = _pick(rng, templates)
    sql = "SELECT reason, COUNT(*) AS num_appointments FROM appointments GROUP BY reason ORDER BY num_appointments DESC"
    return [Example(q, sql, "group_by", "medium", ["appointments"])]


def gen_departments_more_than_n_doctors(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "Which departments have more than {k} doctors?",
        "List departments with more than {k} doctors assigned.",
    ]
    out = []
    for k in _sample(rng, [2, 3, 4, 5, 6, 8, 10], n):
        q = _pick(rng, templates).format(k=k)
        sql = (
            "SELECT dep.department_name, COUNT(*) AS num_doctors FROM doctors d "
            "JOIN departments dep ON d.department_id = dep.department_id "
            f"GROUP BY dep.department_name HAVING COUNT(*) > {k} ORDER BY num_doctors DESC"
        )
        out.append(Example(q, sql, "group_by_having_join", "hard", ["doctors", "departments"]))
    return out


def gen_avg_payment_by_department_via_appointment(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "What is the average appointment payment amount for each department?",
        "Show the average payment amount per department for appointments.",
    ]
    q = _pick(rng, templates)
    sql = (
        "SELECT dep.department_name, AVG(a.payment_amount) AS avg_payment "
        "FROM appointments a "
        "JOIN doctors d ON a.doctor_id = d.doctor_id "
        "JOIN departments dep ON d.department_id = dep.department_id "
        "GROUP BY dep.department_name ORDER BY avg_payment DESC"
    )
    return [Example(q, sql, "group_by_3join", "hard", ["appointments", "doctors", "departments"])]


def gen_max_min_medical_record_metric(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    metrics = [
        ("weight_kg", "weight", "MAX"), ("weight_kg", "weight", "MIN"),
        ("height_cm", "height", "MAX"), ("height_cm", "height", "MIN"),
        ("temperature_f", "temperature", "MAX"), ("temperature_f", "temperature", "MIN"),
    ]
    out = []
    for col, label, agg in _sample(rng, metrics, n):
        word = "highest" if agg == "MAX" else "lowest"
        q = f"What is the {word} recorded {label} across all medical records?"
        sql = f"SELECT {agg}({col}) FROM medical_records"
        out.append(Example(q, sql, "aggregation", "easy", ["medical_records"]))
    return out


def gen_sum_appointment_payment_by_doctor(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "What is the total payment amount collected by doctor {last}?",
        "How much money in total did doctor {last} bring in from appointments?",
    ]
    out = []
    for last in _sample(rng, pools.doctor_last_names, n):
        q = _pick(rng, templates).format(last=last)
        sql = (
            "SELECT SUM(a.payment_amount) FROM appointments a "
            "JOIN doctors d ON a.doctor_id = d.doctor_id "
            f"WHERE d.last_name = '{last}'"
        )
        out.append(Example(q, sql, "aggregation_join", "medium", ["appointments", "doctors"]))
    return out


def gen_distinct_surgery_types(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "What are the distinct types of surgery performed?",
        "List all unique surgery types on record.",
        "Show the different kinds of surgeries performed at the hospital.",
    ]
    q = _pick(rng, templates)
    sql = "SELECT DISTINCT surgery_type FROM surgery_records"
    return [Example(q, sql, "distinct", "easy", ["surgery_records"])]


def gen_top_n_patients_by_surgery_count(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "Which {k} patients have undergone the most surgeries?",
        "Show the top {k} patients by number of surgeries.",
    ]
    out = []
    for k in _sample(rng, [3, 5, 10], n):
        q = _pick(rng, templates).format(k=k)
        sql = (
            "SELECT p.patient_id, p.first_name, p.last_name, COUNT(*) AS num_surgeries "
            "FROM surgery_records s JOIN patients p ON s.patient_id = p.patient_id "
            f"GROUP BY p.patient_id, p.first_name, p.last_name ORDER BY num_surgeries DESC LIMIT {k}"
        )
        out.append(Example(q, sql, "group_by_join_limit", "medium", ["surgery_records", "patients"]))
    return out


def gen_surgeries_by_type_and_room_type(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many {stype} surgeries were performed in a {rtype}?",
        "Count {stype} surgeries that took place in a {rtype}.",
    ]
    out = []
    for _ in range(n):
        stype = _pick(rng, pools.surgery_types)
        rtype = _pick(rng, pools.room_types)
        q = _pick(rng, templates).format(stype=stype, rtype=rtype)
        sql = (
            "SELECT COUNT(*) FROM surgery_records s "
            "JOIN rooms r ON s.room_no = r.room_no "
            f"WHERE s.surgery_type = '{stype}' AND r.room_type = '{rtype}'"
        )
        out.append(Example(q, sql, "and_filter_join", "medium", ["surgery_records", "rooms"]))
    return out


def gen_three_join_surgery_patient_department(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "List the department of every patient who had a {stype} surgery.",
        "Which department is associated with each {stype} surgery, by patient?",
    ]
    out = []
    for stype in _sample(rng, pools.surgery_types, n):
        q = _pick(rng, templates).format(stype=stype)
        sql = (
            "SELECT p.first_name, p.last_name, dep.department_name "
            "FROM surgery_records s "
            "JOIN patients p ON s.patient_id = p.patient_id "
            "JOIN doctors doc ON s.surgeon_id = doc.doctor_id "
            "JOIN departments dep ON doc.department_id = dep.department_id "
            f"WHERE s.surgery_type = '{stype}'"
        )
        out.append(Example(q, sql, "four_table_join", "hard", ["surgery_records", "patients", "doctors", "departments"]))
    return out


def gen_left_join_doctors_without_office(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "Which doctors do not have an office assigned?",
        "List doctors with no assigned room.",
        "Show me every doctor who lacks an office number.",
    ]
    q = _pick(rng, templates)
    sql = (
        "SELECT d.doctor_id, d.first_name, d.last_name FROM doctors d "
        "LEFT JOIN rooms r ON d.office_no = r.room_no "
        "WHERE r.room_no IS NULL"
    )
    return [Example(q, sql, "left_join_null", "medium", ["doctors", "rooms"])]


def gen_left_join_patients_without_appointments(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "Which patients have never had an appointment?",
        "List patients with no appointments on record.",
    ]
    q = _pick(rng, templates)
    sql = (
        "SELECT p.patient_id, p.first_name, p.last_name FROM patients p "
        "LEFT JOIN appointments a ON p.patient_id = a.patient_id "
        "WHERE a.appointment_id IS NULL"
    )
    return [Example(q, sql, "left_join_null", "medium", ["patients", "appointments"])]


def gen_subquery_patients_above_avg_payment(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "Which patients paid more than the average appointment payment amount?",
        "List patients whose appointment payment was above the overall average.",
    ]
    q = _pick(rng, templates)
    sql = (
        "SELECT DISTINCT p.patient_id, p.first_name, p.last_name FROM patients p "
        "JOIN appointments a ON p.patient_id = a.patient_id "
        "WHERE a.payment_amount > (SELECT AVG(payment_amount) FROM appointments)"
    )
    return [Example(q, sql, "subquery", "hard", ["patients", "appointments"])]


def gen_subquery_doctors_with_no_surgeries(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "Which doctors have never performed a surgery?",
        "List doctors who are not on record as a surgeon for any surgery.",
    ]
    q = _pick(rng, templates)
    sql = (
        "SELECT doctor_id, first_name, last_name FROM doctors "
        "WHERE doctor_id NOT IN (SELECT surgeon_id FROM surgery_records WHERE surgeon_id IS NOT NULL)"
    )
    return [Example(q, sql, "subquery_not_in", "hard", ["doctors", "surgery_records"])]


def gen_appointments_ordered_by_amount_limit(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "What are the {k} highest-paying appointments?",
        "Show the top {k} appointments by payment amount.",
    ]
    out = []
    for k in _sample(rng, [3, 5, 10], n):
        q = _pick(rng, templates).format(k=k)
        sql = f"SELECT appointment_id, patient_id, doctor_id, payment_amount FROM appointments ORDER BY payment_amount DESC LIMIT {k}"
        out.append(Example(q, sql, "order_by_limit", "easy", ["appointments"]))
    return out


def gen_wards_in_department(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "Which wards belong to the {dept} department?",
        "List the wards in {dept}.",
    ]
    out = []
    for dept in _sample(rng, pools.department_names, n):
        q = _pick(rng, templates).format(dept=dept)
        sql = (
            "SELECT w.ward_no, w.ward_name FROM wards w "
            "JOIN departments dep ON w.department_id = dep.department_id "
            f"WHERE dep.department_name = '{dept}'"
        )
        out.append(Example(q, sql, "join", "easy", ["wards", "departments"]))
    return out


def gen_beds_in_ward(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many beds are there in the {ward}?",
        "Count the beds located in {ward}.",
    ]
    out = []
    for ward in _sample(rng, pools.ward_names, n):
        q = _pick(rng, templates).format(ward=ward)
        sql = (
            "SELECT COUNT(*) FROM beds b "
            "JOIN wards w ON b.ward_no = w.ward_no "
            f"WHERE w.ward_name = '{ward}'"
        )
        out.append(Example(q, sql, "aggregation_join", "medium", ["beds", "wards"]))
    return out


def gen_current_bed_admissions(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "Which patients are currently admitted (not yet discharged) to a bed?",
        "List patients with an ongoing bed admission (no discharge date).",
    ]
    q = _pick(rng, templates)
    sql = (
        "SELECT p.patient_id, p.first_name, p.last_name, br.admission_date FROM bed_records br "
        "JOIN patients p ON br.patient_id = p.patient_id "
        "WHERE br.discharge_date IS NULL"
    )
    return [Example(q, sql, "join_null_filter", "medium", ["bed_records", "patients"])]


def gen_nurses_by_department(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many nurses work in {dept}?",
        "Count nurses assigned to the {dept} department.",
    ]
    out = []
    for dept in _sample(rng, pools.department_names, n):
        q = _pick(rng, templates).format(dept=dept)
        sql = (
            "SELECT COUNT(*) FROM nurses nu "
            "JOIN departments dep ON nu.department_id = dep.department_id "
            f"WHERE dep.department_name = '{dept}'"
        )
        out.append(Example(q, sql, "aggregation_join", "medium", ["nurses", "departments"]))
    return out


def gen_helpers_by_gender_and_department(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many {g} helpers work in {dept}?",
        "Count {g} staff helpers in the {dept} department.",
    ]
    out = []
    gender_word = {"M": "male", "F": "female"}
    for _ in range(n):
        dept = _pick(rng, pools.department_names)
        g = _pick(rng, pools.genders)
        q = _pick(rng, templates).format(g=gender_word.get(g, g), dept=dept)
        sql = (
            "SELECT COUNT(*) FROM helpers h "
            "JOIN departments dep ON h.department_id = dep.department_id "
            f"WHERE h.gender = '{g}' AND dep.department_name = '{dept}'"
        )
        out.append(Example(q, sql, "and_filter_join", "medium", ["helpers", "departments"]))
    return out


def gen_staff_shifts_on_date_range(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many staff shifts were scheduled in {month} {year}?",
        "Count staff shifts that took place during {month} {year}.",
    ]
    months = [
        ("01", "January"), ("02", "February"), ("03", "March"), ("04", "April"),
        ("05", "May"), ("06", "June"), ("07", "July"), ("08", "August"),
        ("09", "September"), ("10", "October"), ("11", "November"), ("12", "December"),
    ]
    out = []
    for mm, mname in _sample(rng, months, n):
        year = rng.choice([2024, 2025])
        q = _pick(rng, templates).format(month=mname, year=year)
        sql = (
            "SELECT COUNT(*) FROM staff_shifts "
            f"WHERE shift_date >= DATE '{year}-{mm}-01' "
            f"AND shift_date < DATE '{year}-{mm}-01' + INTERVAL 1 MONTH"
        )
        out.append(Example(q, sql, "date_filter", "medium", ["staff_shifts"]))
    return out


def gen_medical_records_by_diagnosis(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many patients were diagnosed with {diag}?",
        "Count the medical records with a diagnosis of {diag}.",
    ]
    out = []
    for diag in _sample(rng, pools.diagnoses, n):
        q = _pick(rng, templates).format(diag=diag)
        sql = f"SELECT COUNT(*) FROM medical_records WHERE diagnosis = '{diag}'"
        out.append(Example(q, sql, "aggregation_filter", "easy", ["medical_records"]))
    return out


def gen_medical_records_patient_doctor_diagnosis(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "Which doctor treated each patient diagnosed with {diag}?",
        "For patients diagnosed with {diag}, who was the treating doctor?",
    ]
    out = []
    for diag in _sample(rng, pools.diagnoses, n):
        q = _pick(rng, templates).format(diag=diag)
        sql = (
            "SELECT p.first_name, p.last_name, doc.first_name, doc.last_name "
            "FROM medical_records m "
            "JOIN patients p ON m.patient_id = p.patient_id "
            "JOIN doctors doc ON m.doctor_id = doc.doctor_id "
            f"WHERE m.diagnosis = '{diag}'"
        )
        out.append(Example(q, sql, "three_table_join", "hard", ["medical_records", "patients", "doctors"]))
    return out


def gen_room_records_by_payment_mode(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many room admissions were paid using {mode}?",
        "Count room-based admissions with payment mode {mode}.",
    ]
    out = []
    for mode in _sample(rng, pools.payment_modes, n):
        q = _pick(rng, templates).format(mode=mode)
        sql = f"SELECT COUNT(*) FROM room_records WHERE mode_of_payment = '{mode}'"
        out.append(Example(q, sql, "aggregation_filter", "easy", ["room_records"]))
    return out


def gen_avg_amount_room_records_by_room_type(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "What is the average admission amount for each room type?",
        "Show the average charge for room admissions, broken down by room type.",
    ]
    q = _pick(rng, templates)
    sql = (
        "SELECT r.room_type, AVG(rr.amount) AS avg_amount FROM room_records rr "
        "JOIN rooms r ON rr.room_no = r.room_no "
        "GROUP BY r.room_type ORDER BY avg_amount DESC"
    )
    return [Example(q, sql, "group_by_join", "medium", ["room_records", "rooms"])]


def gen_doctor_by_surgeon_type(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "List all doctors who are a {stype}.",
        "Which doctors specialize as a {stype}?",
    ]
    out = []
    for stype in _sample(rng, pools.surgeon_types, n):
        q = _pick(rng, templates).format(stype=stype)
        sql = f"SELECT doctor_id, first_name, last_name FROM doctors WHERE surgeon_type = '{stype}'"
        out.append(Example(q, sql, "filter", "easy", ["doctors"]))
    return out


def gen_appointments_by_mode_distinct_patients(pools: ValuePools, rng: random.Random, n: int) -> list[Example]:
    templates = [
        "How many distinct patients booked an appointment {mode}?",
        "Count the number of unique patients who made an appointment via {mode}.",
    ]
    out = []
    for mode in _sample(rng, pools.appointment_modes, n):
        q = _pick(rng, templates).format(mode=mode)
        sql = f"SELECT COUNT(DISTINCT patient_id) FROM appointments WHERE mode_of_appointment = '{mode}'"
        out.append(Example(q, sql, "distinct_aggregation", "medium", ["appointments"]))
    return out


TEMPLATE_GENERATORS = [
    gen_count_doctors_by_department,
    gen_list_doctors_by_department,
    gen_patients_by_gender,
    gen_patients_born_between,
    gen_patients_address_like,
    gen_appointments_by_status_and_mode,
    gen_appointments_status_or_status,
    gen_appointments_reason_count_group_by,
    gen_departments_more_than_n_doctors,
    gen_avg_payment_by_department_via_appointment,
    gen_max_min_medical_record_metric,
    gen_sum_appointment_payment_by_doctor,
    gen_distinct_surgery_types,
    gen_top_n_patients_by_surgery_count,
    gen_surgeries_by_type_and_room_type,
    gen_three_join_surgery_patient_department,
    gen_left_join_doctors_without_office,
    gen_left_join_patients_without_appointments,
    gen_subquery_patients_above_avg_payment,
    gen_subquery_doctors_with_no_surgeries,
    gen_appointments_ordered_by_amount_limit,
    gen_wards_in_department,
    gen_beds_in_ward,
    gen_current_bed_admissions,
    gen_nurses_by_department,
    gen_helpers_by_gender_and_department,
    gen_staff_shifts_on_date_range,
    gen_medical_records_by_diagnosis,
    gen_medical_records_patient_doctor_diagnosis,
    gen_room_records_by_payment_mode,
    gen_avg_amount_room_records_by_room_type,
    gen_doctor_by_surgeon_type,
    gen_appointments_by_mode_distinct_patients,
]


def generate_all(con: duckdb.DuckDBPyConnection, seed: int = 42, n_per_template: int = 12) -> list[Example]:
    rng = random.Random(seed)
    pools = ValuePools.load(con)
    examples: list[Example] = []
    for gen_fn in TEMPLATE_GENERATORS:
        examples.extend(gen_fn(pools, rng, n_per_template))
    return examples
