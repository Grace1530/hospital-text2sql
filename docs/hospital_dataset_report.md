# Hospital-Specific Text-to-SQL Dataset Report

- Candidate examples generated: 352
- Verified (executed successfully against DuckDB): 352
- Rejected: 0
- After deduplication: 333
  - duplicate questions removed: 14
  - duplicate SQL removed: 5

## Category distribution

| category | count |
|---|---|
| aggregation_join | 75 |
| join | 40 |
| and_filter_join | 39 |
| aggregation_filter | 26 |
| between_filter | 20 |
| four_table_join | 20 |
| three_table_join | 20 |
| filter | 20 |
| and_filter | 13 |
| date_filter | 12 |
| or_filter | 9 |
| like_filter | 8 |
| group_by_having_join | 7 |
| aggregation | 6 |
| group_by_join_limit | 3 |
| order_by_limit | 3 |
| distinct_aggregation | 3 |
| left_join_null | 2 |
| group_by | 1 |
| group_by_3join | 1 |
| distinct | 1 |
| subquery | 1 |
| subquery_not_in | 1 |
| join_null_filter | 1 |
| group_by_join | 1 |

## Difficulty distribution

| difficulty | count |
|---|---|
| easy | 146 |
| medium | 137 |
| hard | 50 |