"""
Phase 9 — Our own tokenizer: byte-level Byte-Pair Encoding (BPE), trained
from scratch on our own Text-to-SQL corpus.

This is NOT a pretrained tokenizer: the merge rules and vocabulary are
learned entirely from data/splits/train.jsonl (question + schema + SQL
text) by this module's own `train()` method. No external vocabulary or
tokenizer file is loaded from anywhere.

Design:
- Byte-level (operates on UTF-8 bytes, remapped to printable unicode
  symbols the way GPT-2 popularized) so the tokenizer can represent ANY
  input string -- natural language, SQL keywords, table/column names,
  numbers, dates, punctuation, operators, string literals -- with zero
  out-of-vocabulary characters. This matters because we cannot predict
  every table/column name or literal value in advance.
- A lightweight regex pre-tokenizer groups letters, numbers, and runs of
  punctuation/operators into initial chunks before BPE merges are applied
  within each chunk (never across whitespace), which keeps SQL operators
  like `>=`, `<>`, `!=` and identifiers like `department_id` easy for BPE
  to learn as short, meaningful subword units.
- Special tokens: <pad>, <bos>, <eos>, <unk> (the last is kept only for
  interface completeness / defensive decoding; byte-level coverage means
  it is never actually required to encode any string).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Pre-tokenization regex: words, numbers (with optional decimal), runs of
# punctuation/operators, and whitespace, each as separate chunks.
_PRETOKEN_RE = re.compile(
    r"[A-Za-z_]+[A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|[ \t]+|\r?\n|[^\sA-Za-z0-9]+"
)


def _bytes_to_unicode() -> dict[int, str]:
    """
    Reversible mapping from every byte value (0-255) to a printable unicode
    character, so byte sequences can be manipulated as plain Python strings
    during BPE training/merging (the same well-known trick used to make
    byte-level BPE merge-friendly; the mapping itself carries no learned
    information -- only the merges we train below do).
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(
        range(ord("\xae"), ord("\xff") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


def _pretokenize(text: str) -> list[str]:
    return _PRETOKEN_RE.findall(text)


def _word_to_byte_symbols(word: str) -> tuple[str, ...]:
    return tuple(_BYTE_ENCODER[b] for b in word.encode("utf-8"))


def _get_pair_counts(word_freqs: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            counts[pair] = counts.get(pair, 0) + freq
    return counts


def _merge_word(word: tuple[str, ...], pair: tuple[str, str], merged: str) -> tuple[str, ...]:
    out = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, vocab: dict[str, int] | None = None, merges: list[tuple[str, str]] | None = None):
        self.vocab: dict[str, int] = vocab or {}
        self.merges: list[tuple[str, str]] = merges or []
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self._id_to_token = {i: t for t, i in self.vocab.items()}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    @classmethod
    def train(cls, texts: list[str], vocab_size: int) -> "BPETokenizer":
        if vocab_size < 256 + len(SPECIAL_TOKENS):
            raise ValueError("vocab_size too small for byte-level BPE + special tokens")

        word_freqs: dict[tuple[str, ...], int] = {}
        for text in texts:
            for chunk in _pretokenize(text):
                symbols = _word_to_byte_symbols(chunk)
                word_freqs[symbols] = word_freqs.get(symbols, 0) + 1

        vocab_tokens = list(SPECIAL_TOKENS) + [_BYTE_ENCODER[b] for b in range(256)]
        vocab = {tok: i for i, tok in enumerate(vocab_tokens)}
        merges: list[tuple[str, str]] = []

        num_merges = vocab_size - len(vocab)
        for _ in range(num_merges):
            pair_counts = _get_pair_counts(word_freqs)
            if not pair_counts:
                break
            best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            merged_token = best_pair[0] + best_pair[1]
            if merged_token in vocab:
                # Shouldn't happen with correct bookkeeping, but stay safe.
                break
            vocab[merged_token] = len(vocab)
            merges.append(best_pair)

            new_word_freqs: dict[tuple[str, ...], int] = {}
            for word, freq in word_freqs.items():
                new_word = _merge_word(word, best_pair, merged_token)
                new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq
            word_freqs = new_word_freqs

        return cls(vocab=vocab, merges=merges)

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #

    def _bpe_word(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        word = symbols
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            ranked = [(self._merge_rank[p], p) for p in pairs if p in self._merge_rank]
            if not ranked:
                break
            _, best_pair = min(ranked, key=lambda x: x[0])
            word = _merge_word(word, best_pair, best_pair[0] + best_pair[1])
        return word

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.vocab[BOS_TOKEN])
        for chunk in _pretokenize(text):
            symbols = _word_to_byte_symbols(chunk)
            for tok in self._bpe_word(symbols):
                ids.append(self.vocab.get(tok, self.vocab[UNK_TOKEN]))
        if add_eos:
            ids.append(self.vocab[EOS_TOKEN])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = [self._id_to_token.get(i, UNK_TOKEN) for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        byte_chars = "".join(tokens)
        byte_values = bytes(_BYTE_DECODER[c] for c in byte_chars if c in _BYTE_DECODER)
        return byte_values.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.vocab[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.vocab[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK_TOKEN]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=True, indent=2), encoding="utf-8")
        merges_text = "\n".join(f"{a}\t{b}" for a, b in self.merges)
        (directory / "merges.txt").write_text(merges_text, encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "BPETokenizer":
        directory = Path(directory)
        vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        merges_lines = (directory / "merges.txt").read_text(encoding="utf-8").splitlines()
        merges = [tuple(line.split("\t")) for line in merges_lines if line.strip()]
        return cls(vocab=vocab, merges=merges)
