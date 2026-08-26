"""
Phase 9 — Train our own BPE tokenizer on the combined training corpus and
save the resulting vocabulary/merges as project artifacts.

Usage:
    python scripts/train_tokenizer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402
from src.training.dataset import format_source  # noqa: E402

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
TOKENIZER_DIR = PROJECT_ROOT / "checkpoints" / "tokenizer"
VOCAB_SIZE = 8000


def main() -> None:
    texts = []
    with TRAIN_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(format_source(rec["question"], rec["schema"]))
            texts.append(rec["sql"])

    print(f"Training BPE tokenizer on {len(texts)} texts (vocab_size={VOCAB_SIZE})...")
    tokenizer = BPETokenizer.train(texts, vocab_size=VOCAB_SIZE)
    print(f"Trained vocab size: {tokenizer.vocab_size}")

    tokenizer.save(TOKENIZER_DIR)
    print(f"Saved tokenizer to {TOKENIZER_DIR}")

    # Sanity check a few examples.
    for sample in texts[:2] + texts[-2:]:
        ids = tokenizer.encode(sample)
        decoded = tokenizer.decode(ids)
        assert decoded == sample, f"roundtrip mismatch: {sample!r} -> {decoded!r}"
    print("Roundtrip sanity check passed on sample texts.")


if __name__ == "__main__":
    main()
