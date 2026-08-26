"""
Structural parser for the raw Kaggle hospital-management T-SQL dump.

This module NEVER modifies the raw file. It only reads and parses it into
Python data structures so downstream code (database build, inspection
reports) can consume it reproducibly.

Design notes (see docs/raw_dataset_inspection.md for full details):

- The dump is Microsoft SQL Server (T-SQL) syntax: `Create Database`, `GO`
  batch separators, `Time` columns, no backticks.
- Statements are NOT reliably terminated with ';' (e.g. the Doctor INSERT
  block has no trailing ';'). Real T-SQL doesn't require semicolons —
  statement boundaries are keyword-based — so we split on statement-start
  keywords (Create Table / Insert Into / Create Database / Drop Database /
  Use) rather than on ';'.
- Some rows are intentionally disabled by the dataset author using /* ... */
  block comments; these are stripped and correctly excluded.
- No escaped quotes ('') and no semicolons appear inside string literals
  anywhere in this file (verified), which keeps quote-aware tuple/field
  splitting simple and reliable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any


@dataclass
class ColumnDef:
    name: str
    type: str
    is_primary_key: bool = False


@dataclass
class ForeignKeyDef:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class TableDef:
    name: str
    columns: list[ColumnDef] = field(default_factory=list)
    foreign_keys: list[ForeignKeyDef] = field(default_factory=list)
    order: int = 0
    rows: list[list[str]] = field(default_factory=list)  # raw string tokens, in column order


def strip_block_comments(sql: str) -> str:
    return re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)


def strip_line_comments(sql: str) -> str:
    out_lines = []
    for line in sql.split("\n"):
        idx = line.find("--")
        if idx != -1:
            line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


STATEMENT_START_RE = re.compile(
    r"^\s*(Create\s+Table|Insert\s+Into|Create\s+Database|Drop\s+Database|Use)\b",
    re.IGNORECASE | re.MULTILINE,
)


def split_statements(sql: str) -> list[str]:
    """Split into statements based on statement-start keywords, not ';'."""
    sql = re.sub(r"^\s*GO\s*$", "", sql, flags=re.IGNORECASE | re.MULTILINE)
    starts = [m.start() for m in STATEMENT_START_RE.finditer(sql)]
    if not starts:
        return []
    starts.append(len(sql))
    statements = []
    for i in range(len(starts) - 1):
        chunk = sql[starts[i] : starts[i + 1]].strip()
        if chunk.endswith(";"):
            chunk = chunk[:-1].strip()
        if chunk:
            statements.append(chunk)
    return statements


COLTYPE_RE = re.compile(
    r"^\s*(\w+)\s+([A-Za-z]+(?:\s*\(\s*\d+\s*(?:,\s*\d+\s*)?\))?)\s*(Primary Key)?\s*,?\s*$",
    re.IGNORECASE,
)
FK_RE = re.compile(
    r"Foreign Key\s*\(\s*(\w+)\s*\)\s*References\s+(\w+)\s*\(\s*(\w+)\s*\)",
    re.IGNORECASE,
)


def _split_top_level(body: str, opener: str = "(", closer: str = ")") -> list[str]:
    clauses, depth, buf = [], 0, []
    for ch in body:
        if ch == opener:
            depth += 1
            buf.append(ch)
        elif ch == closer:
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            clauses.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        clauses.append("".join(buf))
    return clauses


def parse_create_table(stmt: str, order: int) -> TableDef | None:
    m = re.match(r"\s*Create\s+Table\s+(\w+)\s*\((.*)\)\s*$", stmt, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    name, body = m.group(1), m.group(2)
    table = TableDef(name=name, order=order)
    for clause in _split_top_level(body):
        clause = clause.strip()
        if not clause:
            continue
        fk_m = FK_RE.search(clause)
        if fk_m:
            table.foreign_keys.append(ForeignKeyDef(fk_m.group(1), fk_m.group(2), fk_m.group(3)))
            continue
        col_m = COLTYPE_RE.match(clause)
        if col_m:
            cname, ctype, pk = col_m.group(1), col_m.group(2), col_m.group(3)
            table.columns.append(ColumnDef(name=cname, type=ctype.strip(), is_primary_key=bool(pk)))
        else:
            raise ValueError(f"Unparsed column clause in table {name!r}: {clause!r}")
    return table


def split_top_level_tuples(values_blob: str) -> list[str]:
    tuples, depth, buf, in_string = [], 0, [], False
    for ch in values_blob:
        if ch == "'":
            in_string = not in_string
            buf.append(ch)
            continue
        if in_string:
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
            if depth == 0:
                tuples.append("".join(buf))
                buf = []
        elif depth > 0:
            buf.append(ch)
    return tuples


def split_tuple_fields(tup: str) -> list[str]:
    assert tup.startswith("(") and tup.endswith(")")
    inner = tup[1:-1]
    fields, buf, in_string = [], [], False
    for ch in inner:
        if ch == "'":
            in_string = not in_string
            buf.append(ch)
        elif ch == "," and not in_string:
            fields.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        fields.append("".join(buf).strip())
    return fields


def parse_insert(stmt: str) -> tuple[str, list[list[str]]] | None:
    m = re.match(r"\s*Insert\s+Into\s+(\w+)\s+Values\s*(.*)$", stmt, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    table_name, values_blob = m.group(1), m.group(2)
    tuples = split_top_level_tuples(values_blob)
    rows = [split_tuple_fields(t) for t in tuples]
    return table_name, rows


def parse_raw_sql(path: Path) -> dict[str, TableDef]:
    """Parse the raw dump into an ordered dict of TableDef (schema + raw string rows)."""
    raw_text = Path(path).read_text(encoding="utf-8-sig")
    cleaned = strip_block_comments(strip_line_comments(raw_text))
    statements = split_statements(cleaned)

    tables: dict[str, TableDef] = {}
    order = 0
    for stmt in statements:
        s = stmt.strip()
        if re.match(r"\s*Create\s+Table", s, re.IGNORECASE):
            t = parse_create_table(s, order)
            order += 1
            if t is None:
                raise ValueError(f"Failed to parse CREATE TABLE statement: {s[:120]!r}")
            tables[t.name] = t
        elif re.match(r"\s*Insert\s+Into", s, re.IGNORECASE):
            parsed = parse_insert(s)
            if parsed is None:
                raise ValueError(f"Failed to parse INSERT statement: {s[:120]!r}")
            tname, rows = parsed
            if tname not in tables:
                raise ValueError(f"INSERT into unknown table {tname!r}")
            n_cols = len(tables[tname].columns)
            for r in rows:
                if len(r) != n_cols:
                    raise ValueError(
                        f"Row in table {tname!r} has {len(r)} fields, expected {n_cols}: {r!r}"
                    )
            tables[tname].rows.extend(rows)
        elif re.match(r"\s*(Create\s+Database|Use\s+\w+|Drop\s+Database)", s, re.IGNORECASE):
            continue
        else:
            raise ValueError(f"Unrecognized top-level statement: {s[:120]!r}")
    return tables


def cast_value(raw: str, sql_type: str) -> Any:
    """Convert a raw SQL literal token into a typed Python value."""
    raw = raw.strip()
    if raw.upper() == "NULL":
        return None
    t = sql_type.lower()
    if raw.startswith("'") and raw.endswith("'"):
        raw = raw[1:-1]
        if t == "date":
            return datetime.strptime(raw, "%Y-%m-%d").date()
        if t == "time":
            parts = raw.split(":")
            parts = [int(p) for p in parts]
            while len(parts) < 3:
                parts.append(0)
            return time(*parts[:3])
        return raw
    if t.startswith("int"):
        return int(raw)
    if t.startswith("decimal") or t.startswith("float") or t.startswith("numeric"):
        return float(raw)
    if t in ("char",):
        return raw
    # Unquoted literal fallback (shouldn't normally happen given the dataset).
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def typed_rows(table: TableDef) -> list[dict[str, Any]]:
    """Return rows as list of {column_name: typed_value} dicts."""
    out = []
    for row in table.rows:
        record = {}
        for col, raw_val in zip(table.columns, row):
            record[col.name] = cast_value(raw_val, col.type)
        out.append(record)
    return out
