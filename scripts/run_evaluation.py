"""
Phase 13 — Run the evaluation pipeline for a trained checkpoint against a
data split, producing machine-readable (JSON) and human-readable
(Markdown) reports.

For "hospital" examples, gold/predicted SQL are executed against the real
database/hospital.duckdb. For "general" examples, a fresh in-memory DuckDB
scratch database is rebuilt per-example from its stored `setup_sql`
(the original CREATE TABLE + INSERT context), so execution accuracy can be
measured on both parts of the corpus, not just the hospital slice.

Usage:
    python scripts/run_evaluation.py --checkpoint checkpoints/pipeline_check/last.pt \
        --split data/splits/test.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.general_dataset_loader import _build_scratch_db  # noqa: E402
from src.data.schema_extractor import extract_schema, serialize_schema_text  # noqa: E402
from src.evaluation.evaluate import EvalItem, evaluate_one, summarize  # noqa: E402
from src.inference.generate import Text2SQLInferenceEngine  # noqa: E402

DB_PATH = PROJECT_ROOT / "database" / "hospital.duckdb"
TOKENIZER_DIR = PROJECT_ROOT / "checkpoints" / "tokenizer"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out-prefix", default="docs/evaluation_report")
    args = parser.parse_args()

    engine = Text2SQLInferenceEngine(args.checkpoint, TOKENIZER_DIR)
    hospital_schema = extract_schema(DB_PATH)
    hospital_schema_text = serialize_schema_text(hospital_schema)
    hospital_known_tables = set(hospital_schema["tables"].keys())
    hospital_con = duckdb.connect(str(DB_PATH), read_only=True)

    records = []
    with Path(args.split).open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]

    print(f"Evaluating {len(records)} examples from {args.split} using checkpoint {args.checkpoint}")

    results = []
    t0 = time.time()
    n_skipped = 0
    for i, rec in enumerate(records):
        if rec["source"] == "hospital":
            con = hospital_con
            known_tables = hospital_known_tables
        else:
            con = _build_scratch_db(rec.get("setup_sql", ""))
            if con is None:
                n_skipped += 1
                continue
            known_tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

        predicted = engine.generate_sql(rec["question"], rec["schema"], max_new_tokens=args.max_new_tokens)
        item = EvalItem(
            question=rec["question"],
            gold_sql=rec["sql"],
            predicted_sql=predicted,
            difficulty=rec.get("difficulty", "unknown"),
            known_tables=known_tables,
        )
        results.append(evaluate_one(item, con))

        if con is not hospital_con:
            con.close()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(records)} ({time.time()-t0:.0f}s elapsed)", flush=True)

    hospital_con.close()

    report = summarize(results)
    print(f"\nEvaluated {report.n} examples ({n_skipped} skipped -- scratch DB rebuild failed).")
    print(f"Exact match: {report.exact_match_rate:.3f} | Execution accuracy: {report.execution_accuracy:.3f} | "
          f"Valid SQL: {report.valid_sql_rate:.3f} | Syntax error rate: {report.syntax_error_rate:.3f}")

    out_json = Path(f"{args.out_prefix}.json")
    out_md = Path(f"{args.out_prefix}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_evaluated": report.n,
        "n_skipped": n_skipped,
        "overall": {
            "exact_match_rate": report.exact_match_rate,
            "execution_accuracy": report.execution_accuracy,
            "valid_sql_rate": report.valid_sql_rate,
            "table_accuracy": report.table_accuracy,
            "column_accuracy": report.column_accuracy,
            "join_accuracy": report.join_accuracy,
            "aggregation_accuracy": report.aggregation_accuracy,
            "filter_accuracy": report.filter_accuracy,
            "syntax_error_rate": report.syntax_error_rate,
        },
        "by_difficulty": report.by_difficulty,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Evaluation Report",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Split: `{args.split}` (n={report.n}, skipped={n_skipped})",
        "",
        "## Overall metrics",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k, v in payload["overall"].items():
        md.append(f"| {k} | {v:.3f} |")
    md += ["", "## By difficulty", "", "| difficulty | n | exact_match | execution_acc | valid_sql | syntax_err |", "|---|---|---|---|---|---|"]
    for diff, m in report.by_difficulty.items():
        md.append(f"| {diff} | {m['n']} | {m['exact_match_rate']:.3f} | {m['execution_accuracy']:.3f} | "
                   f"{m['valid_sql_rate']:.3f} | {m['syntax_error_rate']:.3f} |")
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nReports: {out_json} , {out_md}")


if __name__ == "__main__":
    main()
