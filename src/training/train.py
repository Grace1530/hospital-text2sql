"""
Phase 11/12 — Complete, config-driven training pipeline.

Covers: teacher forcing, cross-entropy loss with padding ignored, causal +
padding masks, AdamW optimizer, warmup+inverse-sqrt LR schedule, gradient
clipping, checkpointing, checkpoint loading / resume, validation, and
structured logging. Runs identically on CPU or CUDA (device is selected
automatically, never hard-coded).

Usage:
    python -m src.training.train --config configs/tiny_experiment.yaml
    python -m src.training.train --config configs/base.yaml --resume checkpoints/base/last.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.model.transformer import (
    ModelConfig,
    Seq2SeqTransformer,
    make_decoder_self_mask,
    make_padding_mask,
)
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.dataset import Text2SQLDataset, make_collate_fn
from src.training.scheduler import WarmupInverseSqrtLR
from src.utils.device import get_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, pad_id: int) -> Seq2SeqTransformer:
    m = cfg["model"]
    model_cfg = ModelConfig(
        vocab_size=m["vocab_size"],
        d_model=m["d_model"],
        num_encoder_layers=m["num_encoder_layers"],
        num_decoder_layers=m["num_decoder_layers"],
        num_heads=m["num_heads"],
        d_ff=m["d_ff"],
        dropout=m["dropout"],
        max_seq_len=m["max_seq_len"],
        pad_id=pad_id,
    )
    return Seq2SeqTransformer(model_cfg)


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
    )


def run_batch(model: Seq2SeqTransformer, batch: dict, device: torch.device, pad_id: int) -> torch.Tensor:
    src = batch["src_ids"].to(device)
    dec_in = batch["decoder_input_ids"].to(device)
    labels = batch["labels"].to(device)

    src_mask = make_padding_mask(src, pad_id)
    tgt_mask = make_decoder_self_mask(dec_in, pad_id)
    logits = model(src, dec_in, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
    return compute_loss(logits, labels)


def evaluate(model: Seq2SeqTransformer, loader: DataLoader, device: torch.device, pad_id: int) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss = run_batch(model, batch, device, pad_id)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def save_checkpoint(
    path: Path,
    model: Seq2SeqTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupInverseSqrtLR,
    epoch: int,
    global_step: int,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: Seq2SeqTransformer,
    optimizer: torch.optim.Optimizer | None,
    scheduler: WarmupInverseSqrtLR | None,
    map_location: str,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0)


def train(cfg: dict, resume_path: str | None = None) -> dict:
    t = cfg["training"]
    set_seed(t["seed"])
    device = get_device()

    tokenizer = BPETokenizer.load(Path(cfg["tokenizer"]["dir"]))
    pad_id = tokenizer.pad_id

    train_ds = Text2SQLDataset(cfg["data"]["train_path"], tokenizer, t["max_src_len"], t["max_tgt_len"])
    val_ds = Text2SQLDataset(cfg["data"]["val_path"], tokenizer, t["max_src_len"], t["max_tgt_len"])
    collate = make_collate_fn(pad_id)
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, collate_fn=collate)

    model = build_model(cfg, pad_id).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=t.get("weight_decay", 0.0)
    )
    scheduler = WarmupInverseSqrtLR(optimizer, d_model=cfg["model"]["d_model"], warmup_steps=t["warmup_steps"])

    start_epoch, global_step = 0, 0
    if resume_path:
        start_epoch, global_step = load_checkpoint(Path(resume_path), model, optimizer, scheduler, str(device))
        print(f"Resumed from {resume_path} at epoch={start_epoch}, step={global_step}")

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    log_path = checkpoint_dir / "train_log.jsonl"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_losses": [], "val_losses": [], "device": str(device), "num_parameters": model.num_parameters()}
    print(f"Device: {device} | Parameters: {model.num_parameters():,} | "
          f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")

    with log_path.open("a", encoding="utf-8") as log_f:
        for epoch in range(start_epoch, t["num_epochs"]):
            epoch_start = time.time()
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = run_batch(model, batch, device, pad_id)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                optimizer.step()
                scheduler.step()
                global_step += 1
                running_loss += loss.item()

                if global_step % t["log_every"] == 0:
                    avg = running_loss / t["log_every"]
                    running_loss = 0.0
                    lr = scheduler.get_last_lr()[0]
                    record = {
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": avg,
                        "lr": lr,
                        "device": str(device),
                    }
                    log_f.write(json.dumps(record) + "\n")
                    log_f.flush()
                    print(f"  epoch {epoch} step {global_step} loss {avg:.4f} lr {lr:.6f}")

            val_loss = evaluate(model, val_loader, device, pad_id)
            epoch_time = time.time() - epoch_start
            record = {
                "epoch": epoch,
                "step": global_step,
                "val_loss": val_loss,
                "epoch_time_sec": epoch_time,
                "device": str(device),
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            print(f"Epoch {epoch} done in {epoch_time:.1f}s | val_loss={val_loss:.4f}")

            history["val_losses"].append(val_loss)
            save_checkpoint(checkpoint_dir / f"epoch_{epoch}.pt", model, optimizer, scheduler, epoch + 1, global_step, cfg)
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch + 1, global_step, cfg)

    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    history = train(cfg, resume_path=args.resume)
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
