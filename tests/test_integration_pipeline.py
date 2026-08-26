"""
Phase 18 — End-to-end CPU integration test.

Exercises the full real chain: dataset -> tokenizer -> model -> training ->
checkpoint -> inference -> SQL safety validation -> DuckDB execution, using
the REAL trained tokenizer and the REAL tiny data subset, training a fresh
tiny model for a couple of steps inside the test (fast, self-contained,
doesn't depend on a checkpoint already existing on disk).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TOKENIZER_DIR = PROJECT_ROOT / "checkpoints" / "tokenizer"
TINY_TRAIN = PROJECT_ROOT / "data" / "splits" / "tiny_train.jsonl"
TINY_VAL = PROJECT_ROOT / "data" / "splits" / "tiny_val.jsonl"
DB_PATH = PROJECT_ROOT / "database" / "hospital.duckdb"


def _tiny_cfg(tmp_path: Path) -> dict:
    return {
        "tokenizer": {"dir": str(TOKENIZER_DIR)},
        "data": {"train_path": str(TINY_TRAIN), "val_path": str(TINY_VAL)},
        "model": {
            "vocab_size": 8000,
            "d_model": 32,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "num_heads": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "max_seq_len": 256,
        },
        "training": {
            "seed": 0,
            "batch_size": 4,
            "num_epochs": 2,
            "warmup_steps": 5,
            "grad_clip": 1.0,
            "weight_decay": 0.0,
            "max_src_len": 256,
            "max_tgt_len": 128,
            "log_every": 100,
            "checkpoint_dir": str(tmp_path / "ckpt"),
        },
    }


@pytest.fixture()
def prerequisites_available():
    if not TOKENIZER_DIR.exists():
        pytest.skip("tokenizer not trained yet -- run scripts/train_tokenizer.py")
    if not TINY_TRAIN.exists():
        pytest.skip("tiny dataset not built yet -- run scripts/make_tiny_dataset.py")
    if not DB_PATH.exists():
        pytest.skip("database not built yet -- run scripts/build_database.py")


def test_full_pipeline_dataset_to_execution(prerequisites_available, tmp_path):
    from src.data.schema_extractor import extract_schema, serialize_schema_text
    from src.inference.generate import Text2SQLInferenceEngine
    from src.sql.pipeline import SafeQueryPipeline
    from src.training.train import train as run_training

    cfg = _tiny_cfg(tmp_path)

    # dataset -> tokenizer -> model -> training -> checkpoint
    history = run_training(cfg)
    assert history["num_parameters"] > 0
    assert len(history["val_losses"]) == cfg["training"]["num_epochs"]
    assert all(v == v for v in history["val_losses"])  # no NaNs

    checkpoint_path = Path(cfg["training"]["checkpoint_dir"]) / "last.pt"
    assert checkpoint_path.exists()

    # checkpoint -> inference
    engine = Text2SQLInferenceEngine(checkpoint_path, TOKENIZER_DIR)
    schema = extract_schema(DB_PATH)
    schema_text = serialize_schema_text(schema)
    sql = engine.generate_sql("How many doctors work in Cardiology?", schema_text, max_new_tokens=32)
    assert isinstance(sql, str)  # untrained/undertrained model -> content is not expected to be correct

    # inference -> SQL safety validation -> DuckDB execution (via the same
    # safe pipeline the web app uses -- must never raise, whatever garbage
    # the barely-trained model produces).
    pipeline = SafeQueryPipeline(engine, DB_PATH, schema_text)
    result = pipeline.run("How many doctors work in Cardiology?")
    assert result.question
    assert isinstance(result.is_valid, bool)
    # Whatever happens, the DB must still be intact afterwards.
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    assert con.execute("SELECT COUNT(*) FROM doctors").fetchone()[0] > 0
    con.close()


def test_checkpoint_resume_continues_training(prerequisites_available, tmp_path):
    from src.training.train import train as run_training

    cfg = _tiny_cfg(tmp_path)
    cfg["training"]["num_epochs"] = 1
    run_training(cfg)

    checkpoint_path = Path(cfg["training"]["checkpoint_dir"]) / "last.pt"
    assert checkpoint_path.exists()

    cfg2 = _tiny_cfg(tmp_path)
    cfg2["training"]["num_epochs"] = 2
    history = run_training(cfg2, resume_path=str(checkpoint_path))
    # Should only train the remaining epoch(s), not restart from epoch 0.
    assert len(history["val_losses"]) == 1
