"""
Phase 10 — Our own Transformer, implemented from scratch in PyTorch.

Architecture: encoder-decoder Transformer (Vaswani et al., 2017 style),
chosen over a decoder-only design because:
  - The encoder can attend BIDIRECTIONALLY over "question + schema", which
    suits understanding a fixed input (which tables/columns matter) better
    than causal-only attention would.
  - Cross-attention gives an explicit, inspectable link from each
    generated SQL token back to the input (question/schema) tokens --
    useful for explainability in an academic project.
  - It cleanly separates "understand the input" from "generate SQL",
    matching how the project is described end-to-end.

Every module below (embeddings, positional encoding, multi-head attention,
feed-forward, encoder/decoder layers, masks) is implemented directly on
top of raw `torch.nn` primitives (Linear, LayerNorm, Embedding, Dropout) --
no `nn.Transformer`, `nn.MultiheadAttention`, or any pretrained module is
used anywhere in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512
    pad_id: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


# --------------------------------------------------------------------------- #
# Positional encoding (fixed, sinusoidal -- no learned parameters, and
# generalizes to sequence lengths beyond those seen in training).
# --------------------------------------------------------------------------- #

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# --------------------------------------------------------------------------- #
# Multi-head attention (scaled dot-product), built from Linear layers.
# --------------------------------------------------------------------------- #

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, t, hd)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        query: (b, tq, d_model)   key/value: (b, tk, d_model)
        mask: broadcastable to (b, 1, tq, tk); True/1 = ATTEND, False/0 = MASK OUT.
        """
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (b, h, tq, tk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (b, h, tq, hd)

        b, _, tq, _ = context.shape
        context = context.transpose(1, 2).contiguous().view(b, tq, self.d_model)
        return self.out_proj(context)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# --------------------------------------------------------------------------- #
# Encoder / decoder layers (Pre-LayerNorm for training stability without
# needing careful warmup tuning -- a good fit for a small CPU-dev model).
# --------------------------------------------------------------------------- #

class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.ff = PositionwiseFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, src_mask))
        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.ff = PositionwiseFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, tgt_mask))
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, memory_mask))
        h = self.norm3(x)
        x = x + self.dropout(self.ff(h))
        return x


# --------------------------------------------------------------------------- #
# Full encoder-decoder model
# --------------------------------------------------------------------------- #

class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_encoding = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.embed_scale = math.sqrt(cfg.d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers)])
        self.encoder_norm = nn.LayerNorm(cfg.d_model)

        self.decoder_layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])
        self.decoder_norm = nn.LayerNorm(cfg.d_model)

        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids) * self.embed_scale
        x = self.pos_encoding(x)
        return self.embed_dropout(x)

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        x = self._embed(src_ids)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self._embed(tgt_ids)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.decoder_norm(x)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode(src_ids, src_mask)
        dec_out = self.decode(tgt_ids, memory, tgt_mask, memory_mask)
        return self.output_proj(dec_out)  # (b, t, vocab_size) logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Mask utilities
# --------------------------------------------------------------------------- #

def make_padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(b, t) token ids -> (b, 1, 1, t) boolean mask, True = real token (attend)."""
    return (ids != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(tgt_len: int, device: torch.device) -> torch.Tensor:
    """(1, 1, t, t) boolean mask, True = allowed (j <= i)."""
    mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def make_decoder_self_mask(tgt_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Combine causal + target padding mask -> (b, 1, t, t)."""
    b, t = tgt_ids.shape
    causal = make_causal_mask(t, tgt_ids.device)  # (1,1,t,t)
    pad = (tgt_ids != pad_id).unsqueeze(1).unsqueeze(2)  # (b,1,1,t)
    return causal & pad
