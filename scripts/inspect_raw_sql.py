"""
Phase 1 — Raw dataset inspection.

Parses the raw Kaggle T-SQL dump (data/raw/Hospital_Management_System.sql)
WITHOUT modifying it, and reports on:

- SQL dialect
- tables, columns, data types, primary keys, foreign keys
- row counts per table
- null value usage
- duplicate primary keys
- dangling foreign keys (referential integrity)
- data quality issues

Writes:
- docs/raw_dataset_inspection.json  (machine-readable)
- docs/raw_dataset_inspection.md    (human-readable)

Usage:
    python scripts/inspect_raw_sql.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.raw_sql_parser import parse_raw_sql, typed_rows  # noqa: E402

RAW_SQL_PATH = PROJECT_ROOT / "data" / "raw" / "Hospital_Management_System.sql"
JSON_REPORT_PATH = PROJECT_ROOT / "docs" / "raw_dataset_inspection.json"
MD_REPORT_PATH = PROJECT_ROOT / "docs" / "raw_dataset_inspection.md"


def main() -> None:
    tables = parse_raw_sql(RAW_SQL_PATH)
    print(f"Raw file: {RAW_SQL_PATH}")
    print(f"Parsed {len(tables)} tables.")

    report: dict = {
        "raw_file": str(RAW_SQL_PATH.relative_to(PROJECT_ROOT)),
        "dialect": "Microsoft SQL Server (T-SQL)",
        "num_tables": len(tables),
        "tables": {},
    }
    md_lines = [
        "# Raw Dataset Inspection Report",
        "",
        "Source: `data/raw/Hospital_Management_System.sql` (Kaggle hospital-management dataset, untouched).",
        "",
        "Dialect: **Microsoft SQL Server (T-SQL)** — evidence: `CREATE DATABASE`, `GO` batch separators, "
        "`Time` column type, no backtick identifiers.",
        "",
        "## Data quality findings",
        "",
        "1. **Missing statement terminator.** The `Doctor` INSERT block has no trailing `;` before "
        "`Insert Into Nurse`. T-SQL does not require semicolons (statement boundaries are keyword-based), "
        "so this is valid T-SQL but breaks naive semicolon-based SQL splitters. Our parser "
        "(`src/data/raw_sql_parser.py`) splits on statement-start keywords instead, which handles this "
        "correctly.",
        "2. **Intentionally-disabled rows.** Some INSERT blocks (e.g. `Room`) contain rows wrapped in "
        "`/* ... */` block comments — the dataset author disabled some generated rows. These are correctly "
        "excluded by comment stripping.",
        "3. **Typo'd column names** carried over from the source: `Nurse.conatct_No` (should be "
        "`contact_No`), `RoomRecords.admisson_ID` (should be `admission_ID`), `Appointment.appoIntment_Id` "
        "(inconsistent casing). These are normalized in the clean schema.",
        "4. **Inconsistent PK naming across near-duplicate tables**: `BedRecords.admission_Id` vs "
        "`RoomRecords.admisson_ID` — same concept, different raw names. Normalized to `admission_id` in "
        "both (still separate tables/ID spaces, not merged).",
        "",
        "## Tables",
        "",
    ]

    for name, table in tables.items():
        rows = typed_rows(table)
        col_names = [c.name for c in table.columns]
        pk_cols = [c.name for c in table.columns if c.is_primary_key]

        pk_col = pk_cols[0] if pk_cols else col_names[0]
        pk_values = [r[pk_col] for r in rows]
        dup_pk = {v: c for v, c in Counter(pk_values).items() if c > 1}

        null_counts = Counter()
        for r in rows:
            for cname, val in r.items():
                if val is None:
                    null_counts[cname] += 1

        print(f"TABLE {name}: {len(rows)} rows, {len(table.columns)} cols, "
              f"{len(table.foreign_keys)} FKs, dup_pk={len(dup_pk)}")

        report["tables"][name] = {
            "order": table.order,
            "columns": [
                {"name": c.name, "type": c.type, "is_primary_key": c.is_primary_key}
                for c in table.columns
            ],
            "foreign_keys": [
                {"column": fk.column, "ref_table": fk.ref_table, "ref_column": fk.ref_column}
                for fk in table.foreign_keys
            ],
            "row_count": len(rows),
            "primary_key_column": pk_col,
            "duplicate_primary_keys": dup_pk,
            "null_value_columns": dict(null_counts),
            "sample_row": {k: str(v) for k, v in rows[0].items()} if rows else None,
        }

        md_lines.append(f"### `{name}` ({len(rows)} rows)")
        md_lines.append("")
        md_lines.append("| column | type | notes |")
        md_lines.append("|---|---|---|")
        for c in table.columns:
            notes = []
            if c.is_primary_key:
                notes.append("PRIMARY KEY")
            fk = next((fk for fk in table.foreign_keys if fk.column == c.name), None)
            if fk:
                notes.append(f"FK -> {fk.ref_table}.{fk.ref_column}")
            if null_counts.get(c.name):
                notes.append(f"{null_counts[c.name]} NULLs")
            md_lines.append(f"| {c.name} | {c.type} | {', '.join(notes)} |")
        md_lines.append("")

    # ---- Referential integrity (dangling foreign keys) ----
    print("\nReferential integrity check:")
    md_lines.append("## Referential integrity")
    md_lines.append("")
    md_lines.append("| table.column | -> ref_table.ref_column | dangling (non-null, no match) |")
    md_lines.append("|---|---|---|")
    integrity_report = []
    typed_cache = {name: typed_rows(t) for name, t in tables.items()}
    for name, table in tables.items():
        rows = typed_cache[name]
        for fk in table.foreign_keys:
            ref_rows = typed_cache.get(fk.ref_table, [])
            ref_pk_col = next(
                (c.name for c in tables[fk.ref_table].columns if c.is_primary_key),
                fk.ref_column,
            )
            ref_values = {r[ref_pk_col] for r in ref_rows}
            dangling = [
                r[fk.column] for r in rows if r[fk.column] is not None and r[fk.column] not in ref_values
            ]
            status = f"{len(dangling)}"
            if dangling:
                status += f" (e.g. {dangling[:3]})"
            print(f"  {name}.{fk.column} -> {fk.ref_table}.{fk.ref_column}: dangling={len(dangling)}")
            md_lines.append(f"| {name}.{fk.column} | {fk.ref_table}.{fk.ref_column} | {status} |")
            integrity_report.append(
                {
                    "table": name,
                    "column": fk.column,
                    "ref_table": fk.ref_table,
                    "ref_column": fk.ref_column,
                    "dangling_count": len(dangling),
                }
            )
    report["referential_integrity"] = integrity_report

    JSON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    MD_REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nJSON report: {JSON_REPORT_PATH}")
    print(f"Markdown report: {MD_REPORT_PATH}")


if __name__ == "__main__":
    main()
