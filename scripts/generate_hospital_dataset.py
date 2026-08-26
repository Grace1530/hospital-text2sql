"""
Phase 5 — Generate, verify, and save the hospital-specific Text-to-SQL dataset.

Usage:
    python scripts/generate_hospital_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hospital_nl2sql_generator import generate_all  # noqa: E402
from src.data.verify_examples import deduplicate, verify_examples  # noqa: E402

DB_PATH = PROJECT_ROOT / "database" / "hospital.duckdb"
OUT_PATH = PROJECT_ROOT / "data" / "datasets" / "hospital_generated.jsonl"
REPORT_PATH = PROJECT_ROOT / "docs" / "hospital_dataset_report.md"

SEED = 42
N_PER_TEMPLATE = 20


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    known_tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

    print("Generating candidate examples from templates...")
    candidates = generate_all(con, seed=SEED, n_per_template=N_PER_TEMPLATE)
    print(f"  generated {len(candidates)} candidate examples")

    print("Verifying every example by executing it against the real database...")
    verified, rejected = verify_examples(con, candidates, known_tables)
    print(f"  verified: {len(verified)}   rejected: {len(rejected)}")

    print("Deduplicating (exact question / exact SQL)...")
    deduped, dedup_stats = deduplicate(verified)
    print(f"  {dedup_stats}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(deduped):
            record = {
                "id": f"hosp_{i:05d}",
                "question": ex.question,
                "sql": ex.sql,
                "category": ex.category,
                "difficulty": ex.difficulty,
                "tables": ex.tables,
                "result_row_count": ex.result_row_count,
                "source": ex.source,
            }
            f.write(json.dumps(record) + "\n")

    category_counts = Counter(ex.category for ex in deduped)
    difficulty_counts = Counter(ex.difficulty for ex in deduped)
    rejection_reasons = Counter(r.reason.split(":")[0] for r in rejected)

    report_lines = [
        "# Hospital-Specific Text-to-SQL Dataset Report",
        "",
        f"- Candidate examples generated: {len(candidates)}",
        f"- Verified (executed successfully against DuckDB): {len(verified)}",
        f"- Rejected: {len(rejected)}",
        f"- After deduplication: {len(deduped)}",
        f"  - duplicate questions removed: {dedup_stats['duplicate_questions_removed']}",
        f"  - duplicate SQL removed: {dedup_stats['duplicate_sql_removed']}",
        "",
        "## Category distribution",
        "",
        "| category | count |",
        "|---|---|",
    ]
    for cat, cnt in category_counts.most_common():
        report_lines.append(f"| {cat} | {cnt} |")
    report_lines += ["", "## Difficulty distribution", "", "| difficulty | count |", "|---|---|"]
    for diff, cnt in difficulty_counts.most_common():
        report_lines.append(f"| {diff} | {cnt} |")
    if rejection_reasons:
        report_lines += ["", "## Rejection reasons", "", "| reason | count |", "|---|---|"]
        for reason, cnt in rejection_reasons.most_common():
            report_lines.append(f"| {reason} | {cnt} |")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    con.close()

    print(f"\nSaved {len(deduped)} verified hospital-specific examples to {OUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
