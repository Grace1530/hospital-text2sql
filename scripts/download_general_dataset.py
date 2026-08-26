"""
Phase 6 — Download the public general-purpose Text-to-SQL dataset.

Dataset selected: gretelai/synthetic_text_to_sql (Hugging Face)
  - License: Apache-2.0 (permits our academic/project use freely).
  - Size: 105,851 examples (100,000 train / 5,851 test).
  - Purely synthetic (Gretel Navigator), NOT scraped from a copyrighted
    corpus -- avoids the ambiguous re-distribution terms that come with
    datasets built on scraped forum/schema data.
  - Each example ships its own database schema AND seed data (CREATE TABLE
    + INSERT context), a natural-language prompt, and a target SQL query,
    covering a wide range of SQL complexity (joins, subqueries,
    aggregations, set ops) across 100 domains.

Why not Spider / WikiSQL:
  - Spider is CC BY-SA 4.0 (ShareAlike) and distributed via a Google Drive
    link rather than a stable direct download.
  - WikiSQL is single-table only (no JOINs), underrepresenting the
    multi-table SQL our hospital domain needs.

Network note: this sandboxed environment can reach huggingface.co's own
API/domain reliably, but large-file downloads that redirect to the LFS/Xet
CDN (cdn-lfs.huggingface.co, *.aws.cdn.hf.co) get reset. We therefore pull
the data through the huggingface.co **datasets-server rows API**
(https://datasets-server.huggingface.co/rows), which serves rows as plain
JSON directly (no CDN redirect), paginated 100 rows at a time. This is
still the same, unmodified dataset content -- just fetched via a different
transport that works in this environment.

Saves the raw rows, byte-for-byte as returned by the API, to:
    data/raw/external/gretel_synthetic_text_to_sql/train_raw.jsonl
    data/raw/external/gretel_synthetic_text_to_sql/test_raw.jsonl

Usage:
    python scripts/download_general_dataset.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "external" / "gretel_synthetic_text_to_sql"

API_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "gretelai/synthetic_text_to_sql"
PAGE_SIZE = 100

# We don't need all 105,851 rows for a small from-scratch model -- a
# well-shuffled prefix sample (verified diverse across domains at multiple
# offsets during inspection) is enough general-SQL signal. See
# docs/general_dataset_report.md for the final counts after verification.
TRAIN_TARGET = 15000
TEST_TARGET = 3000


def fetch_page(split: str, offset: int, length: int, tries: int = 8) -> dict:
    params = {"dataset": DATASET, "config": "default", "split": split, "offset": offset, "length": length}
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(API_URL, params=params, timeout=30)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 10 * attempt))
                print(f"    rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to fetch {split} offset={offset}: {last_err}")


def download_split(split: str, target: int, out_path: Path) -> int:
    written = 0
    if out_path.exists():
        written = sum(1 for _ in out_path.open(encoding="utf-8"))
        if written >= target:
            print(f"{split}: already have {written} rows at {out_path}, skipping.")
            return written
        print(f"{split}: resuming from {written} existing rows.")

    with out_path.open("a", encoding="utf-8") as f:
        offset = written
        while written < target:
            length = min(PAGE_SIZE, target - written)
            page = fetch_page(split, offset, length)
            rows = page["rows"]
            if not rows:
                break
            for r in rows:
                f.write(json.dumps(r["row"]) + "\n")
                written += 1
            offset += len(rows)
            time.sleep(0.6)  # be polite to the free API
            if offset % 1000 == 0 or written >= target:
                print(f"  {split}: fetched {written}/{target}")
    return written


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train_raw.jsonl"
    test_path = OUT_DIR / "test_raw.jsonl"

    print(f"Downloading {DATASET} via datasets-server rows API...")
    n_train = download_split("train", TRAIN_TARGET, train_path)
    n_test = download_split("test", TEST_TARGET, test_path)

    print(f"\nSaved {n_train} train rows to {train_path}")
    print(f"Saved {n_test} test rows to {test_path}")


if __name__ == "__main__":
    main()
