# HANDOFF REPORT — CPU Development Phase Complete

**Status: GPU-READY HANDOFF STATE reached. Serious model training has NOT
been performed. STOP before GPU training, per project instructions.**

This report is the single source of truth for what has and has not been
done. Every number below was produced by actually running the referenced
script/test on this machine — nothing here is estimated or fabricated.

---

## 1. Project structure

```
data/raw/Hospital_Management_System.sql   Kaggle dataset (untouched)
data/raw/external/                        downloaded general dataset (gitignored, re-downloadable)
data/processed/                           schema.json / schema.txt (regenerable)
data/datasets/                            hospital_generated.jsonl, general_verified.jsonl
data/splits/                              train.jsonl / val.jsonl / test.jsonl / tiny_*.jsonl
database/hospital.duckdb                  clean DB (gitignored, regenerable via scripts/build_database.py)
src/{data,tokenizer,model,training,evaluation,inference,sql,utils}/
scripts/                                  one script per pipeline stage
configs/{tiny_experiment,pipeline_check,base}.yaml
checkpoints/{tokenizer,tiny_experiment,pipeline_check}/
web/{app.py,templates/index.html}
tests/                                    115 tests, all passing
docs/                                     generated reports + docs/design_decisions.md
```

## 2. Database status

- Built from the raw T-SQL dump via `scripts/build_database.py` (fully
  reproducible — the DuckDB file is deleted and rebuilt from scratch every
  run).
- **14 tables, 13,543 rows total**, zero dangling foreign keys, zero
  duplicate primary keys (verified in `docs/raw_dataset_inspection.md`
  BEFORE the clean schema was designed, and re-verified against the built
  database in `tests/test_database.py`).
- 13/13 database validation tests pass (table/column existence, PK
  uniqueness, FK integrity, 2/3/4-table joins, LEFT JOIN, GROUP BY/HAVING,
  subqueries, date filtering).

| clean table | rows |
|---|---|
| departments | 31 | rooms | 391 | wards | 63 | beds | 500 |
| doctors | 400 | nurses | 500 | helpers | 1,100 | patients | 1,500 |
| bed_records | 1,000 | room_records | 1,000 | appointments | 1,000 |
| medical_records | 3,000 | staff_shifts | 2,058 | surgery_records | 1,000 |

## 3. Dataset composition

| source | count | verification method |
|---|---|---|
| Hospital-specific (`data/datasets/hospital_generated.jsonl`) | 333 | every example EXECUTED against the real database |
| General (`gretelai/synthetic_text_to_sql`, Apache-2.0) | 12,746 | every example's schema+seed data rebuilt in a scratch DuckDB and the target SQL EXECUTED against it |
| **Combined, deduplicated** | **13,079** | 0 duplicate questions found across the two sources |

Train/val/test split (80/10/10, stratified by source+difficulty, seed=1337):

| split | total | hospital | general |
|---|---|---|---|
| train | 10,461 | 265 | 10,196 |
| val | 1,306 | 32 | 1,274 |
| test | 1,312 | 36 | 1,276 |

**Leakage check: PASSED — zero questions appear in more than one split**
(`tests/test_dataset_quality.py`, 8/8 passing).

General-dataset verification yield was ~71% (12,746 / 18,000 raw rows);
rejections were genuine quality issues in the synthetic source data
(ambiguous columns, multi-schema-qualified tables, GROUP BY errors),
correctly excluded rather than guessed at — see
`docs/general_dataset_report.md`.

## 4. Tokenizer

- Our own byte-level BPE, trained from scratch on the full train split
  (`scripts/train_tokenizer.py`), **not** a pretrained tokenizer.
- **Vocabulary size: 8,000.**
- Roundtrip-verified (encode -> decode reproduces the original string
  exactly, including unicode and unseen characters) — `tests/test_tokenizer.py`, 8/8 passing.
- Saved to `checkpoints/tokenizer/{vocab.json,merges.txt}`.

## 5. Model architecture

Encoder-decoder Transformer, implemented from scratch on raw `torch.nn`
primitives (`src/model/transformer.py`) — see `docs/design_decisions.md`
for the full rationale.

| parameter | value |
|---|---|
| Vocabulary size | 8,000 |
| d_model | 256 |
| Encoder / decoder layers | 4 / 4 |
| Attention heads | 8 |
| Feed-forward dim | 1,024 |
| Dropout | 0.1 |
| Max sequence length | 512 |
| **Total parameters** | **9,421,824** |
| Positional encoding | fixed sinusoidal |
| Normalization | Pre-LayerNorm |
| Weight tying | embedding <-> output projection |

Architecture correctness verified by `tests/test_transformer.py` (9/9
passing), including a causal-mask test that confirms changing a future
token cannot change earlier positions' outputs.

## 6. CPU experiments actually run (NOT real training)

### 6a. Tiny pipeline-verification experiment (`configs/tiny_experiment.yaml`)

- Model: 277,504 params (d_model=32, 1+1 layers). Data: 40 train / 10 val examples.
- 5 epochs, ran in ~1 second total.
- **Train loss: 8.49 -> 3.80** (confirms gradients flow and loss decreases).
- Val loss: 6.60 -> 6.05 (expected to plateau/overfit — 40 examples, not a real training signal).
- Checkpoint saved and reloaded successfully; `Text2SQLInferenceEngine` loaded it and generated a string.

### 6b. Real-model-scale pipeline check (`configs/pipeline_check.yaml`)

- **The actual target model (9,421,824 params, d_model=256)**, on a
  500-example / 100-val real-data subset, 2 epochs.
- **Train loss: 7.41 -> 3.87.** Val loss: 4.58 -> 4.03.
- Epoch times: **59.3s and 69.8s** (500 examples, batch_size=16).
- **Extrapolated full-dataset epoch time on this CPU: ~20-25 minutes**
  (10,461 examples). At `configs/base.yaml`'s 20 epochs, that is
  **6.5-8+ hours of CPU time** — exactly the "serious training" this
  project's instructions say must NOT be run on the CPU dev machine.
  This is the concrete evidence for why GPU training is required, not
  just an assumption.
- Resume-from-checkpoint verified to continue at the correct epoch, not
  restart from zero (`tests/test_integration_pipeline.py`).

### 6c. Evaluation pipeline dry run

Ran `scripts/run_evaluation.py` against the pipeline-check checkpoint (2
epochs, 500 examples) on 40 test examples:

| metric | value |
|---|---|
| exact_match_rate | 0.000 |
| execution_accuracy | 0.000 |
| valid_sql_rate | 0.000 |
| syntax_error_rate | 1.000 |

**This is expected and correctly reported, not a bug.** A model trained
for 2 epochs on 500 examples cannot generate valid SQL yet. What this run
proves is that the evaluation pipeline itself works end-to-end (all 9
metrics computed, broken down by difficulty, JSON+Markdown reports
produced) — see `docs/evaluation_report_pipeline_check.md`.

### 6d. Live web application smoke test

Started the real `uvicorn` server pointing at the pipeline-check
checkpoint and hit it with real HTTP requests:
- `GET /api/status` -> confirmed model + database loaded.
- `POST /api/generate` -> the undertrained model produced degenerate
  output (`"SELECT FROM FROM FROM ..."`), and **the safety validator
  correctly rejected it** (`is_valid: false`, syntax error) rather than
  attempting to execute it.
- `POST /api/execute` with a real safe query (`SELECT COUNT(*) FROM
  doctors`) -> executed correctly, returned `400`.
- `POST /api/execute` with `DROP TABLE doctors` -> **rejected**, database
  confirmed intact afterward.

This demonstrates the full safety-critical path works correctly
regardless of model quality — the dangerous-operation defense does not
depend on the model being good.

## 7. Evaluation pipeline status

Implemented (`src/evaluation/evaluate.py`): exact match, execution
accuracy, valid-SQL rate, table/column/join/aggregation/filter accuracy,
syntax-error rate, all broken down by difficulty. 8/8 unit tests passing
against the real hospital database with hand-crafted gold/predicted pairs
covering every metric's true/false branches. General-dataset examples are
evaluated by rebuilding their exact scratch database from stored
`setup_sql`, so execution accuracy is measurable on both corpus halves.

## 8. Inference pipeline status

`src/inference/generate.py`: loads tokenizer + model config + checkpoint,
generates SQL via greedy decoding (default) or beam search — no external
LLM call anywhere. Verified via the integration test and the live web
server smoke test above.

## 9. SQL validator / safety status

`src/sql/safety.py`: sqlglot-AST-based SELECT-only enforcement + keyword
denylist (string-literal-masked to avoid false positives — a real bug
found on real data and fixed, see `docs/design_decisions.md`) + unknown-
table rejection + single-statement enforcement. 13/13 unit tests passing,
plus 7/7 tests on the full safe-execution pipeline (`test_sql_pipeline.py`)
confirming dangerous SQL is rejected before ever reaching DuckDB and the
database remains provably intact afterward.

## 10. Website status

FastAPI + vanilla HTML/JS (`web/app.py`, `web/templates/index.html`):
question input -> Generate SQL (shows SQL + validation badge) -> Execute
Query (shows results table, row count, execution time, errors). 6/6
endpoint tests passing via `TestClient`, plus the live-server smoke test
in §6d above.

## 11. Test results

**115 / 115 tests passing** (`pytest tests/`), covering: raw SQL parsing
(including the real missing-semicolon bug), database build/validation,
schema extraction, SQL safety, schema filtering, tokenizer, Transformer
architecture (shapes/masks/gradients/determinism/device/save-load),
training dataset/collate, LR scheduler, evaluation metrics, inference
pipeline, safe query pipeline, web app, dataset quality/leakage, and one
true end-to-end integration test that trains a real (tiny) model and runs
it through the entire chain to DuckDB execution.

## 12. GPU readiness status

`scripts/gpu_handoff_check.py`: **6/6 checks pass.** Automatic device
detection works, model/checkpoint are device-independent
(`map_location`-based loading verified), no hard-coded paths in configs,
no hardware-specific package pins in `requirements.txt`.

**GPU training has NOT been performed on this machine** (no CUDA device
present — `torch.cuda.is_available() == False`, confirmed by
`docs/gpu_readiness_check.md`). No results claiming GPU training or a
trained model's real-world accuracy exist anywhere in this repository.

## 13. Known limitations

- No trained model yet — everything in §6 is pipeline verification, not
  training progress. Expect low-to-zero SQL validity until real training
  on the NVIDIA machine.
- General-dataset examples carry synthetic, non-hospital data (schema-only
  seed data), so they teach SQL syntax/composition, not hospital facts.
- Full-schema conditioning (all 14 tables, every example) is appropriate
  at this schema size (~90 columns total) but would need a retrieval step
  (a simple one is already implemented and tested in
  `src/data/schema_filter.py` but not wired into the default pipeline) if
  the schema grew much larger.
- Evaluation's execution-accuracy metric compares result sets ignoring row
  order (standard practice), which can slightly over-credit queries
  missing an `ORDER BY` the gold query has.

## 14. Exact steps to move to the NVIDIA laptop

Full detail in `docs/gpu_training_guide.md`. Summary:

1. Copy the project (excluding `.venv/`).
2. `python -m venv .venv && .venv\Scripts\activate && pip install --upgrade pip`
3. Install CUDA PyTorch FIRST:
   `pip install torch --index-url https://download.pytorch.org/whl/cu121`
   (match the CUDA version actually installed on that machine)
4. `pip install -r requirements.txt`
5. Verify: `python -c "from src.utils.device import device_report; import json; print(json.dumps(device_report(), indent=2))"`
   — must show `"cuda_available": true` before proceeding.
6. If data/checkpoints weren't copied over, regenerate them (commands in
   `README.md` "Reproducing everything from scratch").
7. **Train**: `python -m src.training.train --config configs/base.yaml`
8. **Evaluate**: `python scripts/run_evaluation.py --checkpoint checkpoints/base/last.pt --split data/splits/test.jsonl`
9. **Serve**: `$env:T2SQL_CHECKPOINT="checkpoints/base/last.pt"; uvicorn web.app:app`

### Expected checkpoint/output files after GPU training

- `checkpoints/base/epoch_0.pt` ... `epoch_19.pt`
- `checkpoints/base/last.pt`
- `checkpoints/base/train_log.jsonl` (per-step/per-epoch structured logs)
- `docs/evaluation_report.json` / `.md` (after running the evaluation script)

## 15. What remains to be done after GPU training

1. Run the real training job (`configs/base.yaml`, ~20 epochs, all 10,461
   train examples) and confirm training/val loss actually converges
   (currently only verified to *decrease*, not converge, at tiny scale).
2. Run `scripts/run_evaluation.py` on the FULL test set (1,312 examples,
   not the 40-example smoke sample here) and report real exact-match /
   execution-accuracy numbers.
3. Point the web app at the resulting checkpoint and re-run the same live
   smoke test in §6d, this time expecting valid, correct SQL.
4. Consider hyperparameter iteration (layers, d_model, dropout, LR
   schedule, beam width) now that GPU iteration speed makes that practical
   — none of that was attempted here since it would require exactly the
   long training runs this phase deliberately avoided.
5. Update `README.md` limitations section once real accuracy numbers
   exist, and remove the "not yet trained" caveats.

---

*No performance claims are made in this report beyond what was actually
measured on this machine. GPU training has not occurred.*
