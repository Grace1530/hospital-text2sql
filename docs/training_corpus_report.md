# Training Corpus Report

- Hospital-specific verified examples: 333
- General verified examples: 12746
- Total unique examples after global dedup: 13079 (removed 0 cross-source duplicate questions)
- Split seed: 1337; ratios: train=0.8, val=0.1, test=0.1
- Leakage check: PASSED (no question appears in more than one split)

## Split sizes

| split | count | hospital | general |
|---|---|---|---|
| train | 10461 | 265 | 10196 |
| val | 1306 | 32 | 1274 |
| test | 1312 | 36 | 1276 |

## Difficulty distribution per split

| split | easy | medium | hard |
|---|---|---|---|
| train | 5004 | 4205 | 1252 |
| val | 625 | 525 | 156 |
| test | 627 | 528 | 157 |