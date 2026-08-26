"""
Simple, explainable schema relevance filtering.

The hospital database only has 14 tables (~90 columns total), so the main
training/inference pipeline conditions the Transformer on the FULL schema
for every hospital question -- with a schema this small, feeding the whole
thing is simpler and more reliable than a retrieval step that could hide
the one table the question actually needs. This is a deliberate choice,
not an oversight (see docs/design_decisions.md).

This module provides a lightweight, fully-explainable keyword-overlap
table selector as an OPTIONAL utility for cases where schema size does
matter (e.g. a much larger database, or capping sequence length). It is
unit-tested and available, but not used by default in the main dataset
build.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def select_relevant_tables(
    question: str,
    schema: dict,
    max_tables: int | None = None,
    always_include_referenced_by_fk: bool = True,
) -> list[str]:
    """
    Rank tables by keyword overlap between the question and the table name /
    column names, and return the table names sorted by relevance
    (descending). Ties broken alphabetically for determinism.

    This is intentionally simple (bag-of-words overlap, no ML) so its
    behavior is fully explainable: a table is judged relevant if the
    question shares words with its name or one of its column names.
    """
    q_tokens = _tokenize(question)
    scores: dict[str, int] = {}
    for tname, tinfo in schema["tables"].items():
        table_tokens = _tokenize(tname.replace("_", " "))
        col_tokens: set[str] = set()
        for col in tinfo["columns"]:
            col_tokens |= _tokenize(col["name"].replace("_", " "))
        score = len(q_tokens & table_tokens) * 2 + len(q_tokens & col_tokens)
        scores[tname] = score

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    relevant = [t for t, s in ranked if s > 0]

    if always_include_referenced_by_fk:
        expanded = set(relevant)
        for rel in schema.get("relationships", []):
            if rel["from_table"] in expanded:
                expanded.add(rel["to_table"])
        relevant = [t for t in schema["tables"] if t in expanded]

    if not relevant:
        relevant = sorted(schema["tables"].keys())

    if max_tables is not None:
        relevant = relevant[:max_tables]
    return relevant
