"""
Phase 14 — Reusable inference pipeline.

Loads OUR tokenizer + OUR model architecture + OUR checkpoint and generates
SQL from a natural-language question and a schema text block. This NEVER
calls an external LLM/API -- generation is 100% local, using only the
Transformer implemented in src/model/transformer.py.

Output SQL is NOT executed here -- callers must pass it through
src/sql/safety.py before running it against DuckDB (see src/sql/pipeline.py
for the end-to-end safe pipeline used by the web app).
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.model.transformer import (
    ModelConfig,
    Seq2SeqTransformer,
    make_causal_mask,
    make_padding_mask,
)
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.dataset import format_source
from src.utils.device import get_device


class Text2SQLInferenceEngine:
    def __init__(self, checkpoint_path: str | Path, tokenizer_dir: str | Path, device: torch.device | None = None):
        self.device = device or get_device()
        self.tokenizer = BPETokenizer.load(Path(tokenizer_dir))

        checkpoint = torch.load(Path(checkpoint_path), map_location=str(self.device))
        model_cfg_dict = checkpoint["config"]["model"]
        model_cfg = ModelConfig(
            vocab_size=model_cfg_dict["vocab_size"],
            d_model=model_cfg_dict["d_model"],
            num_encoder_layers=model_cfg_dict["num_encoder_layers"],
            num_decoder_layers=model_cfg_dict["num_decoder_layers"],
            num_heads=model_cfg_dict["num_heads"],
            d_ff=model_cfg_dict["d_ff"],
            dropout=0.0,  # no dropout at inference
            max_seq_len=model_cfg_dict["max_seq_len"],
            pad_id=self.tokenizer.pad_id,
        )
        self.model = Seq2SeqTransformer(model_cfg).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.max_seq_len = model_cfg.max_seq_len

    @torch.no_grad()
    def _encode_source(self, question: str, schema_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        src_text = format_source(question, schema_text)
        src_ids = self.tokenizer.encode(src_text, add_bos=True, add_eos=True)[: self.max_seq_len]
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        src_mask = make_padding_mask(src_tensor, self.tokenizer.pad_id)
        return src_tensor, src_mask

    @torch.no_grad()
    def generate_sql(self, question: str, schema_text: str, max_new_tokens: int = 128) -> str:
        """Greedy decoding: simple, deterministic, and fast enough for a small model."""
        src_tensor, src_mask = self._encode_source(question, schema_text)
        memory = self.model.encode(src_tensor, src_mask)

        generated = [self.tokenizer.bos_id]
        for _ in range(max_new_tokens):
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)
            causal = make_causal_mask(tgt_tensor.size(1), self.device)
            logits = self.model.decode(tgt_tensor, memory, causal, src_mask)
            next_token_logits = self.model.output_proj(logits[:, -1, :])
            next_id = int(torch.argmax(next_token_logits, dim=-1).item())
            generated.append(next_id)
            if next_id == self.tokenizer.eos_id:
                break

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_sql_beam(
        self, question: str, schema_text: str, max_new_tokens: int = 128, beam_width: int = 4
    ) -> str:
        """Beam search decoding -- typically higher quality than greedy at extra compute cost."""
        src_tensor, src_mask = self._encode_source(question, schema_text)
        memory = self.model.encode(src_tensor, src_mask)

        beams: list[tuple[list[int], float]] = [([self.tokenizer.bos_id], 0.0)]
        finished: list[tuple[list[int], float]] = []

        for _ in range(max_new_tokens):
            if not beams:
                break
            candidates = []
            for seq, score in beams:
                tgt_tensor = torch.tensor([seq], dtype=torch.long, device=self.device)
                causal = make_causal_mask(tgt_tensor.size(1), self.device)
                logits = self.model.decode(tgt_tensor, memory, causal, src_mask)
                next_logits = self.model.output_proj(logits[:, -1, :])
                log_probs = torch.log_softmax(next_logits, dim=-1).squeeze(0)
                topk = torch.topk(log_probs, beam_width)
                for tok_id, tok_logp in zip(topk.indices.tolist(), topk.values.tolist()):
                    candidates.append((seq + [tok_id], score + tok_logp))

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = []
            for seq, score in candidates[:beam_width]:
                if seq[-1] == self.tokenizer.eos_id:
                    finished.append((seq, score))
                else:
                    beams.append((seq, score))
            if len(finished) >= beam_width:
                break

        all_candidates = finished if finished else beams
        best_seq, _ = max(all_candidates, key=lambda x: x[1] / max(1, len(x[0])))
        return self.tokenizer.decode(best_seq, skip_special_tokens=True).strip()
