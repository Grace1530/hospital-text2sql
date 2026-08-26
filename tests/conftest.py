import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

DB_PATH = PROJECT_ROOT / "database" / "hospital.duckdb"


@pytest.fixture(scope="session")
def db_path() -> Path:
    if not DB_PATH.exists():
        pytest.skip(f"Database not built yet: {DB_PATH}. Run scripts/build_database.py first.")
    return DB_PATH


@pytest.fixture()
def con(db_path):
    import duckdb

    connection = duckdb.connect(str(db_path), read_only=True)
    yield connection
    connection.close()
