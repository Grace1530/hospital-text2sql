# Evaluation Report

- Checkpoint: `checkpoints/base/epoch_13.pt`
- Split: `data/splits/test.jsonl` (n=1312, skipped=0)

## Overall metrics

| metric | value |
|---|---|
| exact_match_rate | 0.074 |
| execution_accuracy | 0.278 |
| valid_sql_rate | 0.622 |
| table_accuracy | 0.590 |
| column_accuracy | 0.542 |
| join_accuracy | 0.895 |
| aggregation_accuracy | 0.785 |
| filter_accuracy | 0.595 |
| syntax_error_rate | 0.056 |

## By difficulty

| difficulty | n | exact_match | execution_acc | valid_sql | syntax_err |
|---|---|---|---|---|---|
| easy | 627 | 0.124 | 0.345 | 0.648 | 0.035 |
| hard | 157 | 0.006 | 0.119 | 0.535 | 0.153 |
| medium | 528 | 0.034 | 0.236 | 0.617 | 0.051 |