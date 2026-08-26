"""
PyTorch Dataset for the combined Text-to-SQL training corpus.

Encoder input format (one plain-text string, tokenized by our own BPE
tokenizer): "QUESTION: {question}\nSCHEMA:\n{schema}"
Decoder target: the SQL string.

Batches are dynamically padded to the longest example in the batch (not a
fixed max length every time), which keeps CPU experiments fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.tokenizer.bpe_tokenizer import BPETokenizer


def format_source(question: str, schema: str) -> str:
    return f"QUESTION: {question}\nSCHEMA:\n{schema}"


class Text2SQLDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: BPETokenizer,
        max_src_len: int,
        max_tgt_len: int,
    ):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.records = []
        with Path(jsonl_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        src_text = format_source(rec["question"], rec["schema"])
        src_ids = self.tokenizer.encode(src_text, add_bos=True, add_eos=True)[: self.max_src_len]
        tgt_ids = self.tokenizer.encode(rec["sql"], add_bos=True, add_eos=True)[: self.max_tgt_len]
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "id": rec.get("id", str(idx)),
        }


def make_collate_fn(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        src_lens = [len(b["src_ids"]) for b in batch]
        tgt_lens = [len(b["tgt_ids"]) for b in batch]
        max_src = max(src_lens)
        max_tgt = max(tgt_lens)

        src_batch = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
        tgt_batch = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            src_batch[i, : len(b["src_ids"])] = b["src_ids"]
            tgt_batch[i, : len(b["tgt_ids"])] = b["tgt_ids"]

        # Teacher forcing: decoder input is target shifted right (drop last
        # token), label is target shifted left (drop first token, i.e. BOS).
        decoder_input = tgt_batch[:, :-1]
        labels = tgt_batch[:, 1:].clone()
        labels[labels == pad_id] = -100  # ignore_index for cross-entropy

        return {
            "src_ids": src_batch,
            "decoder_input_ids": decoder_input,
            "labels": labels,
            "ids": [b["id"] for b in batch],
        }

    return collate
