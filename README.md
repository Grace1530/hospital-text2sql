# Hospital Natural-Language-to-SQL System Using a Transformer Trained From Scratch

A web application where a user asks a natural-language question about a
hospital database, and a **Transformer trained entirely from random
initialization** (our own architecture, our own tokenizer, our own
training pipeline -- no pretrained LLM, no external API) generates the
corresponding SQL query, which is validated as read-only before running
against a clean DuckDB database.

```
Natural-language question
        |
Relevant hospital database schema
        |
Our Transformer (trained from scratch)
        |
Generated SQL
        |
SQL safety validation (read-only enforcement)
        |
Clean hospital DuckDB database
        |
Query results
        |
Web interface
```

> **Status: CPU-development phase complete, GPU-ready handoff state.**
> Serious model training has **not** been performed yet -- see
> `HANDOFF_REPORT.md` for exactly what has and has not been done, and
> `docs/gpu_training_guide.md` for how to run real training on an NVIDIA
> machine.

## Project structure

```
data/
  raw/Hospital_Management_System.sql   Kaggle dataset (untouched, source of truth)
  raw/external/                        downloaded general Text-to-SQL dataset (gitignored)
  processed/                           extracted schema (JSON + text)
  datasets/                            verified hospital-specific + general examples
  splits/                              train/val/test (deduplicated, leakage-checked)
database/
  hospital.duckdb                      clean DuckDB database (rebuildable, gitignored)
src/
  data/            raw SQL parser, schema mapping, schema extractor, dataset generators
  tokenizer/       from-scratch byte-level BPE tokenizer
  model/           from-scratch encoder-decoder Transformer (PyTorch)
  training/        dataset/collate, LR scheduler, training loop
  evaluation/      exact-match / execution-accuracy / etc. evaluation pipeline
  inference/       checkpoint -> SQL generation (greedy + beam search)
  sql/             safety validator, end-to-end safe query pipeline
  utils/           automatic CPU/CUDA device selection
scripts/           one script per pipeline stage (see "Reproducing everything" below)
configs/           YAML hyperparameter configs (tiny experiment, pipeline check, full/base)
checkpoints/       tokenizer + model checkpoints (gitignored except structure)
web/               FastAPI app + HTML/CSS/JS frontend
tests/             pytest suite (unit + integration), 90+ tests
docs/              generated reports (dataset, database, evaluation) + design decisions
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

This installs the CPU build of PyTorch by default. For the NVIDIA machine,
see `docs/gpu_training_guide.md` for installing a CUDA build first.

## Reproducing everything from scratch

Every artifact below is regenerated deterministically from
`data/raw/Hospital_Management_System.sql` (never modified) plus a
downloaded public dataset (Apache-2.0 licensed, re-downloadable).

```powershell
python scripts/inspect_raw_sql.py            # Phase 1: raw dataset inspection report
python scripts/build_database.py             # Phase 2: build clean DuckDB database
pytest tests/test_database.py                # Phase 3: validate the database
python scripts/extract_schema.py             # Phase 4: machine-readable schema
python scripts/generate_hospital_dataset.py  # Phase 5: hospital-specific Text-to-SQL corpus
python scripts/download_general_dataset.py   # Phase 6: download general Text-to-SQL dataset
python scripts/process_general_dataset.py    # Phase 6/7: verify + normalize general dataset
python scripts/build_training_corpus.py      # Phase 7/8: combine + split train/val/test
python scripts/train_tokenizer.py            # Phase 9: train our BPE tokenizer
python scripts/analyze_token_lengths.py      # sizes max_src_len/max_tgt_len from real data
python scripts/make_tiny_dataset.py          # Phase 11: tiny subset for pipeline verification
python -m src.training.train --config configs/tiny_experiment.yaml   # Phase 11
python -m src.training.train --config configs/base.yaml              # Phase 12 (GPU machine)
python scripts/run_evaluation.py --checkpoint <ckpt> --split data/splits/test.jsonl  # Phase 13
python scripts/gpu_handoff_check.py          # Phase 20
pytest                                        # full test suite
```

## Running the web app

```powershell
$env:T2SQL_CHECKPOINT = "checkpoints/pipeline_check/last.pt"   # or checkpoints/base/last.pt after GPU training
uvicorn web.app:app --reload
```

Then open `http://127.0.0.1:8000`. Ask a question, click **Generate SQL**
(the model output + read-only validation status is shown), then
**Execute Query** to run it against the hospital database.

## The dataset

- **Raw source**: Kaggle hospital-management SQL Server dump, 14 tables,
  13,543 rows total, verified with **zero dangling foreign keys and zero
  duplicate primary keys** (see `docs/raw_dataset_inspection.md`).
- **Hospital-specific Text-to-SQL examples**: ~330 examples generated from
  parametrized templates using real values from the live database, every
  one executed against the real DuckDB file to confirm correctness
  (`docs/hospital_dataset_report.md`).
- **General Text-to-SQL examples**: ~12,700 examples from
  `gretelai/synthetic_text_to_sql` (Apache-2.0, purely synthetic), each
  verified by rebuilding its exact schema+seed data in a scratch DuckDB and
  executing the target SQL (`docs/general_dataset_report.md`).
- **Combined corpus**: ~13,000 examples, globally deduplicated by question,
  split 80/10/10 into train/val/test with **zero cross-split question
  leakage** (`docs/training_corpus_report.md`).

See `docs/design_decisions.md` for why this dataset, why this schema
design, and every other non-obvious choice made along the way.

## The model

Encoder-decoder Transformer, implemented from scratch on raw `torch.nn`
primitives (no `nn.Transformer`, no pretrained weights):

| | |
|---|---|
| Vocabulary | 8,000 (our own byte-level BPE, trained on our corpus) |
| d_model | 256 |
| Encoder / decoder layers | 4 / 4 |
| Attention heads | 8 |
| Feed-forward dim | 1024 |
| Max sequence length | 512 |
| Parameters | 9,421,824 |
| Positional encoding | fixed sinusoidal |
| Normalization | Pre-LayerNorm |

Full architecture and rationale in `src/model/transformer.py` and
`docs/design_decisions.md`.

## SQL safety

Generated SQL is **never** executed directly. It passes through
`src/sql/safety.py`, which: rejects non-SELECT statements (parsed with
`sqlglot`), rejects a denylist of dangerous keywords (as a second,
independent check), rejects multiple stacked statements, and rejects
references to unknown tables. The DuckDB connection used for execution is
additionally opened `read_only=True`. See `tests/test_sql_safety.py` and
`tests/test_sql_pipeline.py`.

## Limitations (current CPU-development state)

- The model has only been trained for a handful of epochs on small
  subsets, purely to verify the pipeline (see `HANDOFF_REPORT.md`). It is
  **not** expected to generate correct SQL reliably yet -- that requires
  the real training run on the NVIDIA GPU machine.
- General-dataset examples are schema-only (no real hospital data), so
  they teach SQL syntax/composition, not hospital domain facts.
- Full-schema conditioning (all 14 tables every time) is a deliberate
  choice appropriate at this schema size; it would not scale unchanged to
  a database with hundreds of tables (see `docs/design_decisions.md`).

## Future work

- Real training run on the NVIDIA GPU machine using `configs/base.yaml`.
- Hyperparameter search (layers, d_model, dropout, LR schedule) once GPU
  iteration speed makes that practical.
- Expand the hospital-specific template corpus with harder multi-join and
  correlated-subquery examples.
- Consider beam-search-width tuning and length normalization once a
  properly trained checkpoint exists to evaluate against.
