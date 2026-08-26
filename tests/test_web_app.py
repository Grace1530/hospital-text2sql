import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "database" / "hospital.duckdb"


class FakeEngine:
    def __init__(self, fixed_sql: str):
        self.fixed_sql = fixed_sql

    def generate_sql(self, question: str, schema_text: str, max_new_tokens: int = 128) -> str:
        return self.fixed_sql


@pytest.fixture()
def client():
    if not DB_PATH.exists():
        pytest.skip("database not built")
    from fastapi.testclient import TestClient

    import web.app as app_module
    from src.sql.pipeline import SafeQueryPipeline

    app_module._pipeline = SafeQueryPipeline(
        FakeEngine("SELECT COUNT(*) FROM doctors"), DB_PATH, schema_text="TABLE doctors\n- doctor_id"
    )
    app_module._load_error = None
    return TestClient(app_module.app)


def test_index_page_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Hospital Text-to-SQL" in res.text


def test_status_endpoint_reports_model_loaded(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    assert res.json()["model_loaded"] is True


def test_generate_endpoint_returns_valid_sql(client):
    res = client.post("/api/generate", json={"question": "How many doctors are there?"})
    data = res.json()
    assert data["is_valid"] is True
    assert "COUNT" in data["generated_sql"]


def test_execute_endpoint_returns_rows(client):
    res = client.post("/api/execute", json={"sql": "SELECT COUNT(*) FROM doctors"})
    data = res.json()
    assert data["row_count"] == 1
    assert data["error"] is None


def test_execute_endpoint_rejects_dangerous_sql(client):
    res = client.post("/api/execute", json={"sql": "DROP TABLE doctors"})
    data = res.json()
    assert data["error"] is not None
    assert "rejected" in data["error"].lower()


def test_generate_endpoint_flags_dangerous_model_output(client):
    import web.app as app_module
    from src.sql.pipeline import SafeQueryPipeline

    app_module._pipeline = SafeQueryPipeline(FakeEngine("DROP TABLE doctors"), DB_PATH, "TABLE doctors\n- doctor_id")
    from fastapi.testclient import TestClient
    c = TestClient(app_module.app)
    res = c.post("/api/generate", json={"question": "delete everything"})
    data = res.json()
    assert data["is_valid"] is False
