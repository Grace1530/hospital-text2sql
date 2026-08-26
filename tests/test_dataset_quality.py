"""
Phase 8 — Automated dataset quality checks on the final train/val/test
splits: no duplicate questions/SQL within a split, no cross-split question
leakage, no obviously invalid records, balanced source representation.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def _load(name):
    path = SPLITS_DIR / f"{name}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not built yet")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def splits():
    return {name: _load(name) for name in ("train", "val", "test")}


def test_splits_are_non_empty(splits):
    for name, records in splits.items():
        assert len(records) > 0, f"{name} split is empty"


def test_no_duplicate_questions_within_a_split(splits):
    for name, records in splits.items():
        questions = [" ".join(r["question"].lower().split()) for r in records]
        counts = Counter(questions)
        dups = {q: c for q, c in counts.items() if c > 1}
        assert not dups, f"{name} has duplicate questions: {list(dups)[:5]}"


def test_no_question_leakage_across_splits(splits):
    q_sets = {
        name: {" ".join(r["question"].lower().split()) for r in records}
        for name, records in splits.items()
    }
    names = list(q_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = q_sets[names[i]] & q_sets[names[j]]
            assert not overlap, f"leakage between {names[i]} and {names[j]}: {list(overlap)[:5]}"


def test_all_records_have_required_fields(splits):
    required = {"id", "question", "schema", "sql", "source", "category", "difficulty"}
    for name, records in splits.items():
        for r in records:
            missing = required - set(r.keys())
            assert not missing, f"{name} record {r.get('id')} missing fields: {missing}"
            assert r["question"].strip(), f"{name} record {r.get('id')} has empty question"
            assert r["sql"].strip(), f"{name} record {r.get('id')} has empty sql"
            assert r["schema"].strip(), f"{name} record {r.get('id')} has empty schema"


def test_source_values_are_known(splits):
    for name, records in splits.items():
        sources = {r["source"] for r in records}
        assert sources <= {"hospital", "general"}, f"{name} has unexpected sources: {sources}"


def test_both_sources_represented_in_every_split(splits):
    for name, records in splits.items():
        sources = {r["source"] for r in records}
        assert "hospital" in sources, f"{name} has no hospital examples"
        assert "general" in sources, f"{name} has no general examples"


def test_difficulty_values_are_known(splits):
    for name, records in splits.items():
        diffs = {r["difficulty"] for r in records}
        assert diffs <= {"easy", "medium", "hard"}, f"{name} has unexpected difficulties: {diffs}"


def test_hospital_examples_reference_real_tables(splits):
    known_tables = {
        "departments", "rooms", "wards", "beds", "doctors", "nurses", "helpers",
        "patients", "bed_records", "room_records", "appointments",
        "medical_records", "staff_shifts", "surgery_records",
    }
    for name, records in splits.items():
        for r in records:
            if r["source"] != "hospital":
                continue
            sql_lower = r["sql"].lower()
            assert any(t in sql_lower for t in known_tables), (
                f"{name} hospital record {r['id']} SQL references no known table: {r['sql']}"
            )
