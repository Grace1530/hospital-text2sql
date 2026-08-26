"""
Phase 16 — Web application.

FastAPI backend serving:
  GET  /                -> HTML page (question input, generate/execute UI)
  POST /api/generate     -> {question} -> generated SQL + validation status
  POST /api/execute      -> {sql} -> query results (re-validated server-side)
  GET  /api/status       -> whether a trained checkpoint is currently loaded

The model checkpoint is loaded once at startup (lazy: if no checkpoint
exists yet, the app still serves the UI and reports a clear "model not
loaded" status instead of crashing -- useful during development before the
first checkpoint exists, and expected to point at the GPU-trained
checkpoint once that becomes available).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema_extractor import extract_schema, serialize_schema_text  # noqa: E402
from src.inference.generate import Text2SQLInferenceEngine  # noqa: E402
from src.sql.pipeline import SafeQueryPipeline  # noqa: E402

DB_PATH = Path(os.environ.get("HOSPITAL_DB_PATH", PROJECT_ROOT / "database" / "hospital.duckdb"))
CHECKPOINT_PATH = Path(os.environ.get("T2SQL_CHECKPOINT", PROJECT_ROOT / "checkpoints" / "tiny_experiment" / "last.pt"))
TOKENIZER_DIR = Path(os.environ.get("T2SQL_TOKENIZER_DIR", PROJECT_ROOT / "checkpoints" / "tokenizer"))

app = FastAPI(title="Hospital Text-to-SQL")

_pipeline: SafeQueryPipeline | None = None
_load_error: str | None = None


def get_pipeline() -> SafeQueryPipeline | None:
    global _pipeline, _load_error
    if _pipeline is not None or _load_error is not None:
        return _pipeline
    try:
        schema = extract_schema(DB_PATH)
        schema_text = serialize_schema_text(schema)
        engine = Text2SQLInferenceEngine(CHECKPOINT_PATH, TOKENIZER_DIR)
        _pipeline = SafeQueryPipeline(engine, DB_PATH, schema_text)
    except Exception as e:  # noqa: BLE001
        _load_error = str(e)
    return _pipeline


class QuestionRequest(BaseModel):
    question: str


class SqlRequest(BaseModel):
    sql: str


@app.get("/api/status")
def status():
    pipeline = get_pipeline()
    return {
        "model_loaded": pipeline is not None,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "database_path": str(DB_PATH),
        "error": _load_error,
    }


@app.post("/api/generate")
def api_generate(req: QuestionRequest):
    pipeline = get_pipeline()
    if pipeline is None:
        return {"error": f"model not loaded: {_load_error}"}
    result = pipeline.generate(req.question)
    return {
        "question": result.question,
        "generated_sql": result.generated_sql,
        "is_valid": result.is_valid,
        "validation_reason": result.validation_reason,
        "generation_time_ms": result.generation_time_ms,
    }


@app.post("/api/execute")
def api_execute(req: SqlRequest):
    pipeline = get_pipeline()
    if pipeline is None:
        return {"error": f"model not loaded: {_load_error}"}
    result = pipeline.execute(req.sql)
    return {
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "execution_time_ms": result.execution_time_ms,
        "error": result.error,
    }


STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).resolve().parent / "templates" / "index.html"
    return html_path.read_text(encoding="utf-8")
