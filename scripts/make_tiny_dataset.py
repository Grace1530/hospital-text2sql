"""
Phase 11 — Carve a deliberately tiny subset out of the real train/val
splits, used ONLY to verify the end-to-end pipeline (dataset -> tokenizer
-> model -> training -> checkpoint -> inference) on this CPU development
machine. This is NOT meant to produce a good model.

Usage:
    python scripts/make_tiny_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

N_TRAIN = 40
N_VAL = 10


def main() -> None:
    with (SPLITS_DIR / "train.jsonl").open(encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f]
    with (SPLITS_DIR / "val.jsonl").open(encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f]

    tiny_train = train_records[:N_TRAIN]
    tiny_val = val_records[:N_VAL]

    with (SPLITS_DIR / "tiny_train.jsonl").open("w", encoding="utf-8") as f:
        for r in tiny_train:
            f.write(json.dumps(r) + "\n")
    with (SPLITS_DIR / "tiny_val.jsonl").open("w", encoding="utf-8") as f:
        for r in tiny_val:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(tiny_train)} tiny_train examples and {len(tiny_val)} tiny_val examples to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
