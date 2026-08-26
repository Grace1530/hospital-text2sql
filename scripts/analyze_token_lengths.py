"""
Analyze tokenized sequence-length distribution on the training split so
max_src_len / max_tgt_len in configs/base.yaml are set from real data,
not guessed.

Usage:
    python scripts/analyze_token_lengths.py
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


def percentile(values: list[int], p: float) -> int:
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]


def main() -> None:
    tokenizer = BPETokenizer.load(TOKENIZER_DIR)
    src_lens, tgt_lens = [], []
    with TRAIN_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            src_lens.append(len(tokenizer.encode(format_source(rec["question"], rec["schema"]))))
            tgt_lens.append(len(tokenizer.encode(rec["sql"])))

    for name, lens in [("source (question+schema)", src_lens), ("target (SQL)", tgt_lens)]:
        print(f"\n{name} token length stats (n={len(lens)}):")
        print(f"  min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f}")
        for p in [0.5, 0.9, 0.95, 0.99, 1.0]:
            print(f"  p{int(p*100)}: {percentile(lens, p)}")


if __name__ == "__main__":
    main()
