"""
Phase 4 — Extract clean schema from database/hospital.duckdb into
data/processed/schema.json, and print the text serialization used for
schema-conditioned Text-to-SQL prompting.

Usage:
    python scripts/extract_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema_extractor import extract_schema, save_schema, serialize_schema_text  # noqa: E402

DB_PATH = PROJECT_ROOT / "database" / "hospital.duckdb"
SCHEMA_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "schema.json"
SCHEMA_TEXT_PATH = PROJECT_ROOT / "data" / "processed" / "schema.txt"


def main() -> None:
    schema = extract_schema(DB_PATH)
    save_schema(schema, SCHEMA_JSON_PATH)
    text = serialize_schema_text(schema)
    SCHEMA_TEXT_PATH.write_text(text, encoding="utf-8")

    print(f"Extracted schema for {len(schema['tables'])} tables, "
          f"{len(schema['relationships'])} foreign-key relationships.")
    for tname, tinfo in schema["tables"].items():
        fks = [c for c in tinfo["columns"] if c["foreign_key"]]
        print(f"  {tname}: {len(tinfo['columns'])} cols, {tinfo['row_count']} rows, "
              f"{len(fks)} FK cols")
    print(f"\nSaved JSON schema to {SCHEMA_JSON_PATH}")
    print(f"Saved text schema to {SCHEMA_TEXT_PATH}")
    print("\n--- Sample text serialization (first 2 tables) ---")
    sample = serialize_schema_text(schema, table_names=sorted(schema["tables"].keys())[:2])
    print(sample)


if __name__ == "__main__":
    main()
