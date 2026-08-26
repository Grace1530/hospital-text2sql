"""
Phase 4 — Machine-readable schema extraction.

Extracts the clean database schema DIRECTLY from the built DuckDB file
(never hand-invented) into a JSON structure, and provides a text
serializer that turns that structure into the schema block fed to the
Transformer alongside each natural-language question.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    is_primary_key: bool
    foreign_key: dict | None = None  # {"table": ..., "column": ...} or None


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: int = 0


def extract_schema(db_path: Path) -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        table_names = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        table_names.sort()

        # Foreign keys via DuckDB's constraint catalog.
        fk_rows = con.execute(
            """
            SELECT
                fk.table_name          AS from_table,
                fk.constraint_column_names AS from_cols,
                pk.table_name          AS to_table,
                pk.constraint_column_names AS to_cols
            FROM duckdb_constraints() fk
            JOIN duckdb_constraints() pk
              ON fk.referenced_table = pk.table_name
             AND fk.constraint_type = 'FOREIGN KEY'
             AND pk.constraint_type = 'PRIMARY KEY'
            """
        ).fetchall() if _has_referenced_table_column(con) else []

        tables: dict[str, TableInfo] = {}
        for tname in table_names:
            pragma_rows = con.execute(f"PRAGMA table_info('{tname}')").fetchall()
            # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
            row_count = con.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
            table = TableInfo(name=tname, row_count=row_count)
            for cid, name, ctype, notnull, dflt, pk in pragma_rows:
                table.columns.append(
                    ColumnInfo(
                        name=name,
                        type=ctype,
                        nullable=not bool(notnull) and not bool(pk),
                        is_primary_key=bool(pk),
                    )
                )
            tables[tname] = table

        # Foreign keys: DuckDB exposes them via duckdb_constraints(); fall back to
        # our own schema_mapping module if the catalog function isn't available
        # in this DuckDB version (kept independent of that module here by
        # re-deriving from information already visible in the live database
        # when possible).
        _attach_foreign_keys_from_duckdb_catalog(con, tables)

        relationships = []
        for tname, tinfo in tables.items():
            for col in tinfo.columns:
                if col.foreign_key:
                    relationships.append(
                        {
                            "from_table": tname,
                            "from_column": col.name,
                            "to_table": col.foreign_key["table"],
                            "to_column": col.foreign_key["column"],
                        }
                    )

        return {
            "database": str(db_path.name),
            "tables": {name: asdict(t) for name, t in tables.items()},
            "relationships": relationships,
        }
    finally:
        con.close()


def _has_referenced_table_column(con) -> bool:
    try:
        con.execute("SELECT referenced_table FROM duckdb_constraints() LIMIT 0")
        return True
    except Exception:
        return False


def _attach_foreign_keys_from_duckdb_catalog(con, tables: dict[str, TableInfo]) -> None:
    try:
        rows = con.execute(
            """
            SELECT table_name, constraint_column_names,
                   referenced_table, referenced_column_names
            FROM duckdb_constraints()
            WHERE constraint_type = 'FOREIGN KEY'
            """
        ).fetchall()
    except Exception:
        rows = []

    for table_name, cols, ref_table, ref_cols in rows:
        if not cols or not ref_cols:
            continue
        local_col = cols[0]
        ref_col = ref_cols[0]
        tinfo = tables.get(table_name)
        if not tinfo:
            continue
        for col in tinfo.columns:
            if col.name == local_col:
                col.foreign_key = {"table": ref_table, "column": ref_col}


def serialize_schema_text(schema: dict, table_names: list[str] | None = None) -> str:
    """
    Render the schema as the plain-text block fed to the Transformer, e.g.:

        TABLE doctors
        - doctor_id
        - department_id
        - first_name
        ...

        TABLE departments
        - department_id
        - department_name

    If `table_names` is given, only those tables are included (schema
    filtering); otherwise the full schema is serialized.
    """
    names = table_names if table_names is not None else sorted(schema["tables"].keys())
    blocks = []
    for tname in names:
        tinfo = schema["tables"].get(tname)
        if tinfo is None:
            continue
        lines = [f"TABLE {tname}"]
        for col in tinfo["columns"]:
            lines.append(f"- {col['name']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def save_schema(schema: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
