# General Text-to-SQL Dataset Report

Source: `gretelai/synthetic_text_to_sql` (Apache-2.0, Hugging Face).

- Raw rows loaded: 18000
- Verified (schema + seed data built AND target SQL executed successfully in DuckDB): 12788
- After deduplication: 12746 (dup questions removed: 3, dup SQL removed: 39)

## Rejection reasons

| reason | count |
|---|---|
| execution_error | 2219 |
| task_type_excluded | 1872 |
| safety | 652 |
| schema_setup_failed | 469 |

## Category (sql_complexity) distribution

| category | count |
|---|---|
| basic SQL | 6110 |
| aggregation | 3281 |
| single join | 1840 |
| subqueries | 656 |
| window functions | 425 |
| multiple_joins | 284 |
| set operations | 150 |

## Difficulty distribution

| difficulty | count |
|---|---|
| easy | 6110 |
| medium | 5121 |
| hard | 1515 |