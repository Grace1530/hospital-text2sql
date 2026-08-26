"""
Phase 7/8 — Combine the hospital-specific and general Text-to-SQL datasets
into one corpus, deduplicate, and create deterministic, leakage-free
train/validation/test splits.

Unified record schema (one JSONL line each):
    {
        "id": str,
        "question": str,
        "schema": str,      # "TABLE x\n- col\n..." text block (see src/data/schema_extractor.py)
        "sql": str,
        "source": "hospital" | "general",
        "category": str,
        "difficulty": "easy" | "medium" | "hard",
    }

Hospital examples are conditioned on the FULL hospital schema (all 14
tables) -- see src/data/schema_filter.py for why full-schema conditioning
was chosen over retrieval at this schema size. General examples carry
their own per-example schema (already exactly the tables needed for that
question, as shipped by the source dataset).

Split strategy: examples are deduplicated globally by normalized question
text FIRST (so no split can ever contain a repeated question), then split
80/10/10 stratified by (source, difficulty) with a fixed seed, so held-out
sets are never touched during training/dev, and hospital vs. general
proportions are preserved across splits.

Usage:
    python scripts/build_training_corpus.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

HOSPITAL_PATH = PROJECT_ROOT / "data" / "datasets" / "hospital_generated.jsonl"
GENERAL_PATH = PROJECT_ROOT / "data" / "datasets" / "general_verified.jsonl"
HOSPITAL_SCHEMA_PATH = PROJECT_ROOT / "data" / "processed" / "schema.txt"

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_PATH = PROJECT_ROOT / "docs" / "training_corpus_report.md"

SEED = 1337
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_question(q: str) -> str:
    return " ".join(q.strip().lower().split())


def build_unified_records() -> list[dict]:
    hospital_rows = load_jsonl(HOSPITAL_PATH)
    general_rows = load_jsonl(GENERAL_PATH)
    hospital_schema_text = HOSPITAL_SCHEMA_PATH.read_text(encoding="utf-8")

    records = []
    for r in hospital_rows:
        records.append(
            {
                "id": f"hospital_{r['id']}",
                "question": r["question"],
                "schema": hospital_schema_text,
                "sql": r["sql"],
                "source": "hospital",
                "category": r["category"],
                "difficulty": r["difficulty"],
            }
        )
    for r in general_rows:
        records.append(
            {
                "id": f"general_{r['id']}",
                "question": r["question"],
                "schema": r["schema"],
                "sql": r["sql"],
                "source": "general",
                "category": r["category"],
                "difficulty": r["difficulty"],
            }
        )
    return records


def deduplicate_global(records: list[dict]) -> tuple[list[dict], int]:
    seen = set()
    out = []
    removed = 0
    for r in records:
        key = normalize_question(r["question"])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(r)
    return out, removed


def stratified_split(records: list[dict], seed: int) -> dict[str, list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["source"], r["difficulty"])].append(r)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for key, items in groups.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    for name in splits:
        rng.shuffle(splits[name])
    return splits


def check_no_leakage(splits: dict[str, list[dict]]) -> list[str]:
    problems = []
    q_by_split = {name: {normalize_question(r["question"]) for r in items} for name, items in splits.items()}
    sql_by_split = {name: {" ".join(r["sql"].split()).lower() for r in items} for name, items in splits.items()}
    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            q_overlap = q_by_split[a] & q_by_split[b]
            if q_overlap:
                problems.append(f"QUESTION leakage between {a} and {b}: {len(q_overlap)} shared questions")
    return problems


def main() -> None:
    print("Loading hospital + general datasets...")
    records = build_unified_records()
    print(f"  combined: {len(records)} records")

    records, n_dup = deduplicate_global(records)
    print(f"  after global dedup: {len(records)} (removed {n_dup} duplicate questions)")

    splits = stratified_split(records, SEED)
    for name, items in splits.items():
        print(f"  {name}: {len(items)}")

    problems = check_no_leakage(splits)
    if problems:
        print("\n!! LEAKAGE DETECTED:")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    else:
        print("\nNo train/val/test question leakage detected.")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        out_path = SPLITS_DIR / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(items)} records to {out_path}")

    # ---- Report ----
    lines = [
        "# Training Corpus Report",
        "",
        f"- Hospital-specific verified examples: {sum(1 for r in records if r['source'] == 'hospital')}",
        f"- General verified examples: {sum(1 for r in records if r['source'] == 'general')}",
        f"- Total unique examples after global dedup: {len(records)} (removed {n_dup} cross-source duplicate questions)",
        f"- Split seed: {SEED}; ratios: train={TRAIN_FRAC}, val={VAL_FRAC}, test={TEST_FRAC}",
        "- Leakage check: PASSED (no question appears in more than one split)" if not problems else "- Leakage check: FAILED",
        "",
        "## Split sizes",
        "",
        "| split | count | hospital | general |",
        "|---|---|---|---|",
    ]
    for name, items in splits.items():
        h = sum(1 for r in items if r["source"] == "hospital")
        g = sum(1 for r in items if r["source"] == "general")
        lines.append(f"| {name} | {len(items)} | {h} | {g} |")

    lines += ["", "## Difficulty distribution per split", "", "| split | easy | medium | hard |", "|---|---|---|---|"]
    for name, items in splits.items():
        c = Counter(r["difficulty"] for r in items)
        lines.append(f"| {name} | {c.get('easy',0)} | {c.get('medium',0)} | {c.get('hard',0)} |")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
