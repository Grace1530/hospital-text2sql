# Evaluation Report

- Checkpoint: `checkpoints/pipeline_check/last.pt`
- Split: `data/splits/test.jsonl` (n=40, skipped=0)

## Overall metrics

| metric | value |
|---|---|
| exact_match_rate | 0.000 |
| execution_accuracy | 0.000 |
| valid_sql_rate | 0.000 |
| table_accuracy | 0.000 |
| column_accuracy | 0.000 |
| join_accuracy | 0.775 |
| aggregation_accuracy | 0.125 |
| filter_accuracy | 0.375 |
| syntax_error_rate | 1.000 |

## By difficulty

| difficulty | n | exact_match | execution_acc | valid_sql | syntax_err |
|---|---|---|---|---|---|
| easy | 19 | 0.000 | 0.000 | 0.000 | 1.000 |
| hard | 3 | 0.000 | 0.000 | 0.000 | 1.000 |
| medium | 18 | 0.000 | 0.000 | 0.000 | 1.000 |