# General Text-to-SQL Dataset Report

Source: `gretelai/synthetic_text_to_sql` (Apache-2.0, Hugging Face).

- Raw rows loaded: 16695
- Verified (schema + seed data built AND target SQL executed successfully in DuckDB): 11860
- After deduplication: 11822 (dup questions removed: 3, dup SQL removed: 35)

## Rejection reasons

| reason | count |
|---|---|
| execution_error | 2071 |
| task_type_excluded | 1732 |
| safety | 602 |
| schema_setup_failed | 430 |

## Category (sql_complexity) distribution

| category | count |
|---|---|
| basic SQL | 5689 |
| aggregation | 3031 |
| single join | 1696 |
| subqueries | 610 |
| window functions | 395 |
| multiple_joins | 264 |
| set operations | 137 |

## Difficulty distribution

| difficulty | count |
|---|---|
| easy | 5689 |
| medium | 4727 |
| hard | 1406 |