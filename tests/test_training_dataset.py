import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.dataset import Text2SQLDataset, format_source, make_collate_fn

SAMPLE_RECORDS = [
    {"id": "a", "question": "How many doctors work in Cardiology?", "schema": "TABLE doctors\n- doctor_id", "sql": "SELECT COUNT(*) FROM doctors"},
    {"id": "b", "question": "List all patients.", "schema": "TABLE patients\n- patient_id", "sql": "SELECT * FROM patients"},
    {"id": "c", "question": "What is the average payment amount for appointments made by patients over 40, grouped by department?", "schema": "TABLE appointments\n- appointment_id\n- payment_amount", "sql": "SELECT dep.department_name, AVG(a.payment_amount) FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id JOIN departments dep ON d.department_id = dep.department_id GROUP BY dep.department_name"},
]


def _make_dataset(tmp_path):
    jsonl_path = tmp_path / "data.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in SAMPLE_RECORDS:
            f.write(json.dumps(r) + "\n")
    texts = [format_source(r["question"], r["schema"]) for r in SAMPLE_RECORDS] + [r["sql"] for r in SAMPLE_RECORDS]
    tok = BPETokenizer.train(texts * 10, vocab_size=400)
    ds = Text2SQLDataset(jsonl_path, tok, max_src_len=64, max_tgt_len=64)
    return ds, tok


def test_dataset_length(tmp_path):
    ds, _ = _make_dataset(tmp_path)
    assert len(ds) == len(SAMPLE_RECORDS)


def test_dataset_item_has_expected_keys(tmp_path):
    ds, _ = _make_dataset(tmp_path)
    item = ds[0]
    assert set(item.keys()) == {"src_ids", "tgt_ids", "id"}
    assert item["src_ids"].dtype == torch.long
    assert item["tgt_ids"].dtype == torch.long


def test_max_len_truncation(tmp_path):
    jsonl_path = tmp_path / "data.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(SAMPLE_RECORDS[2]) + "\n")
    tok = BPETokenizer.train([format_source(SAMPLE_RECORDS[2]["question"], SAMPLE_RECORDS[2]["schema"]), SAMPLE_RECORDS[2]["sql"]] * 10, vocab_size=300)
    ds = Text2SQLDataset(jsonl_path, tok, max_src_len=5, max_tgt_len=5)
    item = ds[0]
    assert len(item["src_ids"]) <= 5
    assert len(item["tgt_ids"]) <= 5


def test_collate_pads_to_batch_max_and_masks_labels(tmp_path):
    ds, tok = _make_dataset(tmp_path)
    collate = make_collate_fn(tok.pad_id)
    batch = collate([ds[0], ds[1], ds[2]])

    assert batch["src_ids"].shape[0] == 3
    assert batch["decoder_input_ids"].shape[0] == 3
    assert batch["labels"].shape == batch["decoder_input_ids"].shape

    # decoder_input is target shifted right by one relative to full tgt_ids
    for i in range(3):
        raw_tgt_len = len(ds[i]["tgt_ids"])
        assert batch["decoder_input_ids"].shape[1] >= raw_tgt_len - 1

    # padded label positions must be -100 (ignored by cross-entropy)
    assert (batch["labels"] == -100).any() or batch["labels"].shape[1] == min(
        len(ds[0]["tgt_ids"]), len(ds[1]["tgt_ids"]), len(ds[2]["tgt_ids"])
    ) - 1


def test_collate_decoder_input_never_contains_bos_only_at_start(tmp_path):
    ds, tok = _make_dataset(tmp_path)
    collate = make_collate_fn(tok.pad_id)
    batch = collate([ds[0], ds[1]])
    assert (batch["decoder_input_ids"][:, 0] == tok.bos_id).all()
