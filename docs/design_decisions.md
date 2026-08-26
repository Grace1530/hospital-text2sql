# Design Decisions

This document records the non-obvious engineering decisions made while
building this project and why, so they don't need to be re-derived later.

## Raw dataset parsing

- The raw dump is **Microsoft SQL Server (T-SQL)** syntax (`CREATE DATABASE`,
  `GO` batch separators, `Time` columns, no backticks).
- T-SQL does not require semicolons between statements (statement
  boundaries are keyword-based). The raw file actually relies on this: the
  `Doctor` INSERT block has **no trailing `;`** before `Insert Into Nurse`.
  A naive semicolon-based statement splitter silently merges the two
  statements together and corrupts both tables' row counts. Our parser
  (`src/data/raw_sql_parser.py`) instead splits on statement-start keywords
  (`Create Table`, `Insert Into`, `Create Database`, `Drop Database`, `Use`),
  which matches how SQL Server itself would parse the file.
- Some rows are intentionally disabled by the dataset author with
  `/* ... */` block comments (e.g. in the `Room` INSERT block). These are
  stripped along with the comment, correctly excluding those rows.
- Verified: no escaped quotes (`''`) and no semicolons inside string
  literals anywhere in the file, which keeps quote-aware tuple/field
  splitting simple and reliable (no generic SQL grammar needed).

## Clean schema design

- Every raw table maps 1:1 to exactly one clean snake_case table (see
  `src/data/schema_mapping.py`). No merges/splits, no canonical-to-raw
  translation layer -- the Transformer generates SQL directly against the
  clean names, and the clean DuckDB file IS the database it runs against.
- `BedRecords.admission_Id` and `RoomRecords.admisson_ID` (typo in the
  raw column name) both become `admission_id`, but the two tables stay
  separate (different admission types / ID spaces in the source data).
- Referential integrity was verified BEFORE designing the clean schema
  (zero dangling foreign keys, zero duplicate primary keys across all 14
  tables, 13,543 total rows) -- the dataset needed no repair, only
  renaming/typing.

## SQL safety validator

- Defense in depth: (1) parse with `sqlglot` and require the root node be
  a SELECT/WITH/UNION/etc; (2) reject any AST node matching
  INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/etc; (3) an independent raw-text
  keyword denylist scan as a backstop in case parsing is ever fooled;
  (4) reject unknown table references against the live schema.
- **Bug found and fixed during dataset generation**: the raw-text keyword
  scan originally ran over the whole SQL string, so a legitimate data
  value like `mode_of_appointment = 'Call'` was flagged as the `CALL`
  keyword. Fixed by masking the contents of `'...'` string literals before
  the keyword regex runs. Covered by regression tests
  (`tests/test_sql_safety.py`).
- CTE aliases (from `WITH x AS (...)`) are excluded from the "unknown
  table" check -- they are not real tables and were originally
  (incorrectly) flagged as such.

## Dataset sourcing

- **Hospital-specific examples**: generated from ~30 parametrized template
  functions (`src/data/hospital_nl2sql_generator.py`), each drawing REAL
  values from the live database (department names, dates, diagnoses, ...),
  with multiple natural-language phrasings per SQL pattern. Every single
  example is executed against the real DuckDB database; only examples that
  execute successfully are kept (`src/data/verify_examples.py`).
- **General-purpose examples**: `gretelai/synthetic_text_to_sql`
  (Hugging Face, Apache-2.0, 105,851 examples, purely synthetic --- not
  scraped from a copyrighted corpus, which avoids the ShareAlike
  constraints that come with Spider). Chosen over WikiSQL (single-table
  only, no JOINs) and Spider (CC BY-SA 4.0, Google-Drive-only
  distribution). See `scripts/download_general_dataset.py` for the license
  citation.
- **Network note**: this sandboxed dev environment can reach
  `huggingface.co`'s own domain but gets connection resets against the
  LFS/Xet CDN redirect target used for direct parquet downloads. Data was
  instead pulled through the `datasets-server.huggingface.co` **rows API**,
  which serves the identical row content as plain JSON with no CDN
  redirect. This is a transport workaround only -- the dataset content is
  unchanged.
- Every general-dataset example ships its own schema + seed data
  (CREATE TABLE + INSERT). Each example is verified by actually building
  that exact mini-database in a fresh in-memory DuckDB connection and
  executing the target SQL against it -- the same "never trust generated
  SQL, always execute it" principle used for the hospital data. About ~70%
  of candidate rows survive verification; rejections are mostly genuine
  quality issues in the synthetic source data (ambiguous column
  references, multi-schema-qualified table names, GROUP BY errors) that we
  correctly exclude rather than "fix" with a guess.

## Schema conditioning: full schema, not retrieval

- The hospital database has only 14 tables (~90 columns total). Feeding
  the model the FULL schema for every question is small (well under a
  few hundred tokens) and strictly simpler/more reliable than a retrieval
  step that could hide the one table a question actually needs -- so full-
  schema conditioning is the default for all hospital training examples
  and at inference time.
- A simple, fully-explainable keyword-overlap table selector is still
  implemented and unit-tested (`src/data/schema_filter.py`) for
  documentation/generality, but is NOT used in the default pipeline. It
  scores tables by word overlap between the question and table/column
  names, then expands the result along foreign-key edges.
- General-dataset examples already come with exactly the schema needed
  for that question (that's what "sql_context" is), so no filtering is
  needed there either.

## Tokenizer

- Byte-level BPE, trained from scratch on our own corpus
  (`src/tokenizer/bpe_tokenizer.py`) -- no pretrained tokenizer or
  vocabulary is loaded from anywhere. Byte-level was chosen specifically
  because we cannot enumerate every table/column name or literal value in
  advance; it guarantees zero out-of-vocabulary characters for any input.

## Model architecture

- Encoder-decoder Transformer (not decoder-only): the encoder attends
  bidirectionally over "question + schema" (useful since the whole input
  is known up front), and cross-attention gives an explicit, inspectable
  link from generated SQL tokens back to specific input tokens --
  valuable for explainability in an academic project.
- Pre-LayerNorm sublayers (norm before self/cross-attention and
  feed-forward, not after) for training stability without needing a
  carefully hand-tuned warmup schedule -- a more forgiving choice for a
  small model trained partly on a CPU.
- Sinusoidal (non-learned) positional encoding: zero extra parameters and
  generalizes beyond the exact lengths seen during training.
- Output projection weights are tied to the input token embedding
  (standard parameter-saving technique, appropriate given the small
  vocabulary/model size here).

## What is explicitly NOT done here

- No pretrained language model or tokenizer anywhere in the final SQL
  generation path.
- No external LLM API calls.
- No canonical-schema-to-raw-schema translation layer.
- No serious/long training run performed on the CPU development machine
  (see HANDOFF_REPORT.md for exactly what WAS run and why).
