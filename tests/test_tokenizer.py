import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer.bpe_tokenizer import BPETokenizer

SAMPLE_TEXTS = [
    "How many doctors work in Cardiology?",
    "SELECT COUNT(*) FROM doctors d JOIN departments dep ON d.department_id = dep.department_id WHERE dep.department_name = 'Cardiology';",
    "TABLE doctors\n- doctor_id\n- department_id\n- first_name\n- last_name",
    "SELECT * FROM patients WHERE date_of_birth BETWEEN '1990-01-01' AND '2000-12-31' AND amount >= 100.50",
    "List patients whose address contains Lahore.",
    "Ünïcödé test: café, naïve, 日本語",
]


def _train_small_tokenizer(vocab_size=300) -> BPETokenizer:
    return BPETokenizer.train(SAMPLE_TEXTS * 5, vocab_size=vocab_size)


def test_train_produces_requested_vocab_size_or_less():
    tok = _train_small_tokenizer(vocab_size=300)
    assert tok.vocab_size <= 300
    assert tok.vocab_size > 256  # at least base bytes + specials


def test_roundtrip_encode_decode_exact():
    tok = _train_small_tokenizer()
    for text in SAMPLE_TEXTS:
        ids = tok.encode(text, add_bos=True, add_eos=True)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == text, f"roundtrip failed for {text!r}: got {decoded!r}"


def test_bos_eos_present():
    tok = _train_small_tokenizer()
    ids = tok.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_no_bos_eos_when_disabled():
    tok = _train_small_tokenizer()
    ids = tok.encode("hello", add_bos=False, add_eos=False)
    assert tok.bos_id not in ids
    assert tok.eos_id not in ids


def test_unseen_characters_still_roundtrip():
    tok = _train_small_tokenizer()
    weird = "🎉 emoji and \t tabs \n newlines"
    ids = tok.encode(weird)
    assert tok.decode(ids) == weird


def test_save_and_load_roundtrip():
    tok = _train_small_tokenizer()
    with tempfile.TemporaryDirectory() as d:
        tok.save(Path(d))
        loaded = BPETokenizer.load(Path(d))
        assert loaded.vocab == tok.vocab
        assert loaded.merges == tok.merges
        for text in SAMPLE_TEXTS:
            assert loaded.decode(loaded.encode(text)) == text


def test_empty_string():
    tok = _train_small_tokenizer()
    ids = tok.encode("", add_bos=True, add_eos=True)
    assert tok.decode(ids) == ""


def test_sql_keywords_tokenize_reasonably_compactly():
    tok = BPETokenizer.train(SAMPLE_TEXTS * 20, vocab_size=500)
    ids = tok.encode("SELECT COUNT(*) FROM doctors", add_bos=False, add_eos=False)
    # With real merges learned, this should be well under one token per byte.
    raw_byte_len = len("SELECT COUNT(*) FROM doctors".encode("utf-8"))
    assert len(ids) < raw_byte_len
