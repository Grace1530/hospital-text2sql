"""
Phase 6/7 — Process the raw general Text-to-SQL dataset into verified,
schema-conditioned examples ready to merge with the hospital dataset.

Usage:
    python scripts/process_general_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.general_dataset_loader import process_row  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "external" / "gretel_synthetic_text_to_sql"
OUT_PATH = PROJECT_ROOT / "data" / "datasets" / "general_verified.jsonl"
REPORT_PATH = PROJECT_ROOT / "docs" / "general_dataset_report.md"


def load_raw(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    all_rows = []
    for split_file in ["train_raw.jsonl", "test_raw.jsonl"]:
        p = RAW_DIR / split_file
        if p.exists():
            rows = load_raw(p)
            print(f"Loaded {len(rows)} rows from {split_file}")
            all_rows.extend(rows)

    if not all_rows:
        print("No raw rows found. Run scripts/download_general_dataset.py first.")
        return

    kept = []
    rejection_reasons = Counter()
    seen_questions = set()
    seen_sql = set()
    dup_q, dup_sql = 0, 0

    for i, row in enumerate(all_rows):
        example, reason = process_row(row)
        if example is None:
            rejection_reasons[reason.split(":")[0]] += 1
            continue
        q_key = example.question.strip().lower()
        sql_key = " ".join(example.sql.split()).lower()
        if q_key in seen_questions:
            dup_q += 1
            continue
        if sql_key in seen_sql:
            dup_sql += 1
            continue
        seen_questions.add(q_key)
        seen_sql.add(sql_key)
        kept.append(example)
        if (i + 1) % 2000 == 0:
            print(f"  processed {i+1}/{len(all_rows)}, kept so far: {len(kept)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(kept):
            record = {
                "id": f"gen_{i:06d}",
                "question": ex.question,
                "sql": ex.sql,
                "schema": ex.schema_text,
                "category": ex.category,
                "difficulty": ex.difficulty,
                "tables": ex.tables,
                "domain": ex.domain,
                "result_row_count": ex.result_row_count,
                "source": ex.source,
            }
            f.write(json.dumps(record) + "\n")

    category_counts = Counter(ex.category for ex in kept)
    difficulty_counts = Counter(ex.difficulty for ex in kept)

    lines = [
        "# General Text-to-SQL Dataset Report",
        "",
        "Source: `gretelai/synthetic_text_to_sql` (Apache-2.0, Hugging Face).",
        "",
        f"- Raw rows loaded: {len(all_rows)}",
        f"- Verified (schema + seed data built AND target SQL executed successfully in DuckDB): {len(kept) + dup_q + dup_sql}",
        f"- After deduplication: {len(kept)} (dup questions removed: {dup_q}, dup SQL removed: {dup_sql})",
        "",
        "## Rejection reasons",
        "",
        "| reason | count |",
        "|---|---|",
    ]
    for reason, cnt in rejection_reasons.most_common():
        lines.append(f"| {reason} | {cnt} |")
    lines += ["", "## Category (sql_complexity) distribution", "", "| category | count |", "|---|---|"]
    for cat, cnt in category_counts.most_common():
        lines.append(f"| {cat} | {cnt} |")
    lines += ["", "## Difficulty distribution", "", "| difficulty | count |", "|---|---|"]
    for diff, cnt in difficulty_counts.most_common():
        lines.append(f"| {diff} | {cnt} |")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nKept {len(kept)} verified general examples -> {OUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
