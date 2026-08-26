"""
Phase 2 — Build the clean DuckDB hospital database from the raw T-SQL dump.

Pipeline:
    data/raw/Hospital_Management_System.sql  (untouched, source of truth)
        -> src/data/raw_sql_parser.py  (structural parse, typed rows)
        -> src/data/schema_mapping.py  (raw -> clean snake_case mapping)
        -> database/hospital.duckdb    (clean, consistent, constrained)

Rerunning this script always produces the same database from the same raw
file (the output file is deleted and rebuilt from scratch every run), so the
clean database is fully reproducible and never hand-edited.

Usage:
    python scripts/build_database.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.raw_sql_parser import parse_raw_sql, typed_rows  # noqa: E402
from src.data.schema_mapping import BUILD_ORDER, TABLE_MAPS_BY_CLEAN_NAME  # noqa: E402

RAW_SQL_PATH = PROJECT_ROOT / "data" / "raw" / "Hospital_Management_System.sql"
DB_PATH = PROJECT_ROOT / "database" / "hospital.duckdb"
REPORT_PATH = PROJECT_ROOT / "docs" / "database_build_report.md"


def build_ddl(clean_name: str) -> str:
    tmap = TABLE_MAPS_BY_CLEAN_NAME[clean_name]
    col_lines = []
    for col in tmap.columns:
        line = f"    {col.clean_name} {col.duckdb_type}"
        if col.clean_name == tmap.primary_key:
            line += " PRIMARY KEY"
        col_lines.append(line)
    for local_col, ref_table, ref_col in tmap.foreign_keys:
        col_lines.append(f"    FOREIGN KEY ({local_col}) REFERENCES {ref_table}({ref_col})")
    body = ",\n".join(col_lines)
    return f"CREATE TABLE {clean_name} (\n{body}\n)"


def main() -> None:
    print(f"Parsing raw SQL: {RAW_SQL_PATH}")
    raw_tables = parse_raw_sql(RAW_SQL_PATH)
    raw_typed = {name: typed_rows(t) for name, t in raw_tables.items()}

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing database at {DB_PATH} (rebuilding from scratch).")

    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA enable_checkpoint_on_shutdown")

    report_lines = [
        "# Database Build Report",
        "",
        f"Built from `data/raw/Hospital_Management_System.sql` into `database/hospital.duckdb`.",
        "",
        "| clean table | raw table | rows loaded | primary key | foreign keys |",
        "|---|---|---|---|---|",
    ]

    total_rows = 0
    for clean_name in BUILD_ORDER:
        tmap = TABLE_MAPS_BY_CLEAN_NAME[clean_name]
        ddl = build_ddl(clean_name)
        con.execute(ddl)

        raw_rows = raw_typed[tmap.raw_name]
        clean_col_names = [c.clean_name for c in tmap.columns]
        raw_col_names = [c.raw_name for c in tmap.columns]

        values = [[row[rc] for rc in raw_col_names] for row in raw_rows]
        placeholders = ", ".join(["?"] * len(clean_col_names))
        insert_sql = f"INSERT INTO {clean_name} ({', '.join(clean_col_names)}) VALUES ({placeholders})"
        con.executemany(insert_sql, values)

        count = con.execute(f"SELECT COUNT(*) FROM {clean_name}").fetchone()[0]
        total_rows += count
        fk_desc = "; ".join(f"{lc}->{rt}.{rc}" for lc, rt, rc in tmap.foreign_keys) or "-"
        print(f"  {clean_name:<20s} <- {tmap.raw_name:<15s} {count:>6d} rows")
        report_lines.append(
            f"| {clean_name} | {tmap.raw_name} | {count} | {tmap.primary_key} | {fk_desc} |"
        )

    print(f"\nTotal rows loaded: {total_rows}")
    report_lines.append("")
    report_lines.append(f"**Total rows loaded: {total_rows}**")

    con.close()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nDatabase written to: {DB_PATH}")
    print(f"Build report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
