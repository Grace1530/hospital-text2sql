"""
SQL safety validation.

The final application is READ-ONLY. Model-generated SQL is NEVER executed
directly — it always passes through this validator first.

Defense in depth:
1. Parse with sqlglot (DuckDB dialect). Reject anything that isn't exactly
   one SELECT-shaped statement (SELECT, or WITH ... SELECT, or a UNION of
   SELECTs).
2. Reject any statement containing a forbidden keyword/node type
   (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, ATTACH, COPY,
   EXPORT, PRAGMA, CALL, SET, ...), even if sqlglot's parse were somehow
   fooled — a plain keyword scan on the raw text as a second, independent
   check.
3. Reject multiple statements stacked with ';'.
4. Reject references to tables outside the known clean schema (defends
   against the model hallucinating a system table).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE",
    "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT", "PRAGMA", "CALL",
    "SET", "GRANT", "REVOKE", "VACUUM", "REINDEX", "MERGE", "REPLACE",
    "LOAD", "INSTALL",
]

# Node types sqlglot may parse these into; any occurrence anywhere in the
# statement tree is rejected outright.
FORBIDDEN_EXP_TYPES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Create,
    exp.Command,  # generic/unclassified statement (covers ATTACH, PRAGMA, etc.)
)

ALLOWED_ROOT_TYPES = (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.With)

_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE
)
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")


def _mask_string_literals(sql: str) -> str:
    """Replace contents of '...' string literals so keyword scans never match data values (e.g. mode='Call')."""
    return _STRING_LITERAL_RE.sub("''", sql)


@dataclass
class ValidationResult:
    is_safe: bool
    reason: str = ""
    normalized_sql: str = ""


def validate_sql_safety(sql: str, known_tables: set[str] | None = None) -> ValidationResult:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return ValidationResult(False, "empty SQL")

    masked = _mask_string_literals(sql)

    if ";" in masked:
        return ValidationResult(False, "multiple statements are not allowed")

    if _FORBIDDEN_RE.search(masked):
        return ValidationResult(False, "forbidden keyword detected")

    try:
        parsed = sqlglot.parse(sql, read="duckdb")
    except Exception as e:  # noqa: BLE001
        return ValidationResult(False, f"SQL syntax error: {e}")

    if len(parsed) != 1 or parsed[0] is None:
        return ValidationResult(False, "expected exactly one statement")

    root = parsed[0]

    if not isinstance(root, ALLOWED_ROOT_TYPES):
        return ValidationResult(False, f"only SELECT statements are allowed, got {type(root).__name__}")

    for node in root.walk():
        node_obj = node[0] if isinstance(node, tuple) else node
        if isinstance(node_obj, FORBIDDEN_EXP_TYPES):
            return ValidationResult(False, f"forbidden operation: {type(node_obj).__name__}")

    if known_tables is not None:
        cte_names = {cte.alias.lower() for cte in root.find_all(exp.CTE)}
        referenced = {t.name.lower() for t in root.find_all(exp.Table)}
        unknown = referenced - {t.lower() for t in known_tables} - cte_names
        if unknown:
            return ValidationResult(False, f"unknown table(s) referenced: {sorted(unknown)}")

    return ValidationResult(True, "ok", normalized_sql=root.sql(dialect="duckdb"))


def extract_referenced_tables(sql: str) -> set[str]:
    try:
        parsed = sqlglot.parse_one(sql, read="duckdb")
    except Exception:  # noqa: BLE001
        return set()
    return {t.name for t in parsed.find_all(exp.Table)}
