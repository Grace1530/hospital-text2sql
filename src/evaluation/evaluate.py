"""
Phase 13 — Evaluation pipeline.

Computes, per prediction (gold_sql, predicted_sql, a DuckDB connection to
execute both against):
  1. Exact SQL match (whitespace/case-normalized string comparison)
  2. Execution accuracy (result sets match, row order ignored)
  3. Valid SQL percentage (parses + passes the read-only safety validator)
  4. Table selection accuracy (same set of referenced tables)
  5. Column selection accuracy (same set of referenced columns)
  6. JOIN accuracy (same set of joined tables)
  7. Aggregation accuracy (same set of aggregate functions used)
  8. Filtering accuracy (same set of columns referenced in WHERE)
  9. Syntax error rate (predicted SQL fails to parse at all)

All metrics are also broken down by difficulty (easy/medium/hard).
This module works on any dataset (hospital or general) -- the caller
supplies the DuckDB connection to execute against per example.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import duckdb
import sqlglot
from sqlglot import exp

from src.sql.safety import validate_sql_safety


@dataclass
class EvalItem:
    question: str
    gold_sql: str
    predicted_sql: str
    difficulty: str = "unknown"
    known_tables: set[str] | None = None


@dataclass
class EvalResult:
    exact_match: bool
    execution_match: bool | None  # None if execution couldn't be attempted (e.g. gold itself fails)
    valid_sql: bool
    table_accuracy: float
    column_accuracy: float
    join_accuracy: float
    aggregation_accuracy: float
    filter_accuracy: float
    syntax_error: bool
    difficulty: str


def _normalize_sql_string(sql: str) -> str:
    return " ".join(sql.strip().lower().split()).rstrip(";")


def _safe_parse(sql: str):
    try:
        return sqlglot.parse_one(sql, read="duckdb")
    except Exception:  # noqa: BLE001
        return None


AGG_FUNCS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


def _extract_tables(tree) -> set[str]:
    if tree is None:
        return set()
    return {t.name.lower() for t in tree.find_all(exp.Table)}


def _extract_columns(tree) -> set[str]:
    if tree is None:
        return set()
    return {c.name.lower() for c in tree.find_all(exp.Column)}


def _extract_aggregations(tree) -> set[str]:
    if tree is None:
        return set()
    found = set()
    for fn in tree.find_all(exp.AggFunc):
        found.add(type(fn).__name__.upper())
    return found


def _extract_where_columns(tree) -> set[str]:
    if tree is None:
        return set()
    cols = set()
    for where in tree.find_all(exp.Where):
        cols |= {c.name.lower() for c in where.find_all(exp.Column)}
    return cols


def _set_accuracy(gold: set, pred: set) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    return len(gold & pred) / len(gold | pred)


def _execute(con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple] | None:
    try:
        return con.execute(sql).fetchall()
    except Exception:  # noqa: BLE001
        return None


def evaluate_one(item: EvalItem, con: duckdb.DuckDBPyConnection) -> EvalResult:
    gold_tree = _safe_parse(item.gold_sql)
    pred_tree = _safe_parse(item.predicted_sql)

    syntax_error = pred_tree is None
    exact_match = _normalize_sql_string(item.gold_sql) == _normalize_sql_string(item.predicted_sql)

    safety = validate_sql_safety(item.predicted_sql, known_tables=item.known_tables)
    valid_sql = safety.is_safe

    execution_match: bool | None = None
    if valid_sql:
        gold_result = _execute(con, item.gold_sql)
        pred_result = _execute(con, item.predicted_sql)
        if gold_result is not None and pred_result is not None:
            execution_match = Counter(map(str, gold_result)) == Counter(map(str, pred_result))
        elif gold_result is None:
            execution_match = None  # gold itself doesn't execute in this context -- not the model's fault
        else:
            execution_match = False

    table_acc = _set_accuracy(_extract_tables(gold_tree), _extract_tables(pred_tree))
    column_acc = _set_accuracy(_extract_columns(gold_tree), _extract_columns(pred_tree))
    join_acc = _set_accuracy(_extract_tables(gold_tree), _extract_tables(pred_tree)) if (
        gold_tree and list(gold_tree.find_all(exp.Join))
    ) else 1.0
    agg_acc = _set_accuracy(_extract_aggregations(gold_tree), _extract_aggregations(pred_tree))
    filter_acc = _set_accuracy(_extract_where_columns(gold_tree), _extract_where_columns(pred_tree))

    return EvalResult(
        exact_match=exact_match,
        execution_match=execution_match,
        valid_sql=valid_sql,
        table_accuracy=table_acc,
        column_accuracy=column_acc,
        join_accuracy=join_acc,
        aggregation_accuracy=agg_acc,
        filter_accuracy=filter_acc,
        syntax_error=syntax_error,
        difficulty=item.difficulty,
    )


@dataclass
class AggregateReport:
    n: int
    exact_match_rate: float
    execution_accuracy: float
    valid_sql_rate: float
    table_accuracy: float
    column_accuracy: float
    join_accuracy: float
    aggregation_accuracy: float
    filter_accuracy: float
    syntax_error_rate: float
    by_difficulty: dict = field(default_factory=dict)


def summarize(results: list[EvalResult]) -> AggregateReport:
    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _summarize_subset(subset: list[EvalResult]) -> dict:
        n = len(subset)
        exec_evaluable = [r for r in subset if r.execution_match is not None]
        return {
            "n": n,
            "exact_match_rate": _avg([1.0 if r.exact_match else 0.0 for r in subset]),
            "execution_accuracy": _avg([1.0 if r.execution_match else 0.0 for r in exec_evaluable]),
            "valid_sql_rate": _avg([1.0 if r.valid_sql else 0.0 for r in subset]),
            "table_accuracy": _avg([r.table_accuracy for r in subset]),
            "column_accuracy": _avg([r.column_accuracy for r in subset]),
            "join_accuracy": _avg([r.join_accuracy for r in subset]),
            "aggregation_accuracy": _avg([r.aggregation_accuracy for r in subset]),
            "filter_accuracy": _avg([r.filter_accuracy for r in subset]),
            "syntax_error_rate": _avg([1.0 if r.syntax_error else 0.0 for r in subset]),
        }

    overall = _summarize_subset(results)
    by_diff: dict[str, dict] = {}
    groups = defaultdict(list)
    for r in results:
        groups[r.difficulty].append(r)
    for diff, subset in groups.items():
        by_diff[diff] = _summarize_subset(subset)

    return AggregateReport(
        n=overall["n"],
        exact_match_rate=overall["exact_match_rate"],
        execution_accuracy=overall["execution_accuracy"],
        valid_sql_rate=overall["valid_sql_rate"],
        table_accuracy=overall["table_accuracy"],
        column_accuracy=overall["column_accuracy"],
        join_accuracy=overall["join_accuracy"],
        aggregation_accuracy=overall["aggregation_accuracy"],
        filter_accuracy=overall["filter_accuracy"],
        syntax_error_rate=overall["syntax_error_rate"],
        by_difficulty=by_diff,
    )
