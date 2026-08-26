import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.transformer import (
    ModelConfig,
    Seq2SeqTransformer,
    make_decoder_self_mask,
    make_padding_mask,
)

VOCAB_SIZE = 50
PAD_ID = 0


def tiny_config(**overrides) -> ModelConfig:
    cfg = ModelConfig(
        vocab_size=VOCAB_SIZE,
        d_model=16,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=32,
        dropout=0.0,
        max_seq_len=32,
        pad_id=PAD_ID,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_forward_output_shape():
    torch.manual_seed(0)
    model = Seq2SeqTransformer(tiny_config())
    src = torch.randint(1, VOCAB_SIZE, (4, 10))
    tgt = torch.randint(1, VOCAB_SIZE, (4, 7))
    src_mask = make_padding_mask(src, PAD_ID)
    tgt_mask = make_decoder_self_mask(tgt, PAD_ID)
    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
    assert logits.shape == (4, 7, VOCAB_SIZE)


def test_causal_mask_blocks_future_tokens():
    torch.manual_seed(0)
    model = Seq2SeqTransformer(tiny_config())
    model.eval()
    src = torch.randint(1, VOCAB_SIZE, (1, 5))
    tgt = torch.randint(1, VOCAB_SIZE, (1, 5))
    src_mask = make_padding_mask(src, PAD_ID)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        tgt_mask_full = make_decoder_self_mask(tgt, PAD_ID)
        out_full = model.decode(tgt, memory, tgt_mask_full, src_mask)

        # Change a FUTURE token (position 4) and re-run; earlier positions'
        # outputs must be unaffected because of the causal mask.
        tgt_changed = tgt.clone()
        tgt_changed[0, 4] = (tgt_changed[0, 4] + 1) % VOCAB_SIZE
        tgt_mask_changed = make_decoder_self_mask(tgt_changed, PAD_ID)
        out_changed = model.decode(tgt_changed, memory, tgt_mask_changed, src_mask)

    assert torch.allclose(out_full[0, :4], out_changed[0, :4], atol=1e-5)


def test_padding_mask_ignored_in_attention_output_for_real_tokens():
    torch.manual_seed(0)
    model = Seq2SeqTransformer(tiny_config())
    model.eval()
    src1 = torch.tensor([[5, 6, 7, PAD_ID, PAD_ID]])
    src2 = torch.tensor([[5, 6, 7, PAD_ID, PAD_ID, PAD_ID, PAD_ID]])  # more padding appended
    tgt = torch.tensor([[3, 4]])

    with torch.no_grad():
        mem1 = model.encode(src1, make_padding_mask(src1, PAD_ID))
        mem2 = model.encode(src2, make_padding_mask(src2, PAD_ID))
        tgt_mask = make_decoder_self_mask(tgt, PAD_ID)
        out1 = model.decode(tgt, mem1, tgt_mask, make_padding_mask(src1, PAD_ID))
        out2 = model.decode(tgt, mem2, tgt_mask, make_padding_mask(src2, PAD_ID))

    assert torch.allclose(out1, out2, atol=1e-4)


def test_gradients_flow_to_all_parameters():
    torch.manual_seed(0)
    model = Seq2SeqTransformer(tiny_config())
    src = torch.randint(1, VOCAB_SIZE, (2, 6))
    tgt = torch.randint(1, VOCAB_SIZE, (2, 5))
    logits = model(src, tgt, make_padding_mask(src, PAD_ID), make_decoder_self_mask(tgt, PAD_ID), make_padding_mask(src, PAD_ID))
    loss = logits.sum()
    loss.backward()
    n_missing = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    assert n_missing == 0


def test_deterministic_with_fixed_seed():
    def build_and_run():
        torch.manual_seed(42)
        model = Seq2SeqTransformer(tiny_config())
        model.eval()
        src = torch.randint(1, VOCAB_SIZE, (2, 6))
        tgt = torch.randint(1, VOCAB_SIZE, (2, 5))
        with torch.no_grad():
            return model(src, tgt, make_padding_mask(src, PAD_ID), make_decoder_self_mask(tgt, PAD_ID), make_padding_mask(src, PAD_ID))

    out1 = build_and_run()
    out2 = build_and_run()
    assert torch.allclose(out1, out2)


def test_weight_tying_between_embedding_and_output_projection():
    model = Seq2SeqTransformer(tiny_config())
    assert model.output_proj.weight is model.token_embedding.weight


def test_parameter_count_is_reported():
    model = Seq2SeqTransformer(tiny_config())
    assert model.num_parameters() > 0


def test_model_moves_to_device_cpu_and_cuda_if_available():
    model = Seq2SeqTransformer(tiny_config())
    model_cpu = model.to("cpu")
    src = torch.randint(1, VOCAB_SIZE, (1, 4))
    tgt = torch.randint(1, VOCAB_SIZE, (1, 3))
    out = model_cpu(src, tgt, make_padding_mask(src, PAD_ID), make_decoder_self_mask(tgt, PAD_ID), make_padding_mask(src, PAD_ID))
    assert out.device.type == "cpu"

    if torch.cuda.is_available():
        model_cuda = model.to("cuda")
        src_c, tgt_c = src.to("cuda"), tgt.to("cuda")
        out_c = model_cuda(src_c, tgt_c, make_padding_mask(src_c, PAD_ID), make_decoder_self_mask(tgt_c, PAD_ID), make_padding_mask(src_c, PAD_ID))
        assert out_c.device.type == "cuda"


def test_save_and_load_state_dict_roundtrip(tmp_path):
    model = Seq2SeqTransformer(tiny_config())
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    model2 = Seq2SeqTransformer(tiny_config())
    model2.load_state_dict(torch.load(path, map_location="cpu"))

    src = torch.randint(1, VOCAB_SIZE, (1, 4))
    tgt = torch.randint(1, VOCAB_SIZE, (1, 3))
    model.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model(src, tgt, make_padding_mask(src, PAD_ID), make_decoder_self_mask(tgt, PAD_ID), make_padding_mask(src, PAD_ID))
        out2 = model2(src, tgt, make_padding_mask(src, PAD_ID), make_decoder_self_mask(tgt, PAD_ID), make_padding_mask(src, PAD_ID))
    assert torch.allclose(out1, out2)
