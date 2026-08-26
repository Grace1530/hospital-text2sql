"""
Phase 20 — GPU handoff validation.

Runs everything that CAN be verified about CUDA-readiness on a machine
that does not have a CUDA GPU, and is explicit and honest about what
cannot be verified here. Never claims GPU training happened if it did not.

Usage:
    python scripts/gpu_handoff_check.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.transformer import ModelConfig, Seq2SeqTransformer  # noqa: E402
from src.utils.device import device_report, get_device  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "docs" / "gpu_readiness_check.md"


def check(name: str, passed: bool, detail: str = "") -> dict:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    return {"name": name, "passed": passed, "detail": detail}


def main() -> None:
    results = []

    dev_report = device_report()
    cuda_available = dev_report["cuda_available"]
    results.append(check(
        "Automatic device detection runs without error",
        True,
        json.dumps(dev_report),
    ))

    device = get_device()
    results.append(check(
        "get_device() returns 'cuda' when available, else 'cpu' (no hard-coding)",
        (device.type == "cuda") == cuda_available,
    ))

    # Model can be constructed and moved to whatever device IS available.
    cfg = ModelConfig(vocab_size=100, d_model=16, num_encoder_layers=1, num_decoder_layers=1, num_heads=2, d_ff=32)
    model = Seq2SeqTransformer(cfg)
    try:
        model.to(device)
        results.append(check(f"Model moves to detected device ({device})", True))
    except Exception as e:  # noqa: BLE001
        results.append(check(f"Model moves to detected device ({device})", False, str(e)))

    # Device-independent checkpoint save/load: save on current device, load
    # with map_location=cpu (always works regardless of what saved it) --
    # this is the exact mechanism used for a GPU-trained checkpoint being
    # loaded for CPU inference/deployment later.
    try:
        with tempfile.TemporaryDirectory() as d:
            ckpt_path = Path(d) / "model.pt"
            torch.save(model.state_dict(), ckpt_path)
            state = torch.load(ckpt_path, map_location="cpu")
            model2 = Seq2SeqTransformer(cfg)
            model2.load_state_dict(state)
        results.append(check("Checkpoint save/load is device-independent (map_location works)", True))
    except Exception as e:  # noqa: BLE001
        results.append(check("Checkpoint save/load is device-independent (map_location works)", False, str(e)))

    # No hard-coded absolute paths tied to this laptop in config files.
    bad_paths = []
    for cfg_file in (PROJECT_ROOT / "configs").glob("*.yaml"):
        text = cfg_file.read_text(encoding="utf-8")
        if "C:\\" in text or "/Users/" in text or "/home/" in text:
            bad_paths.append(str(cfg_file))
    results.append(check(
        "No hard-coded absolute/user-specific paths in configs/*.yaml",
        not bad_paths,
        f"offending files: {bad_paths}" if bad_paths else "all configs use relative paths",
    ))

    # requirements.txt has no OS/hardware-specific pins (check actual
    # dependency lines only -- comments are free to explain the CPU/CUDA
    # distinction in prose, that's not a hard-coded pin).
    req_lines = [
        line for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    bad_pins = [line for line in req_lines if "+cpu" in line.lower() or "+cu1" in line.lower() or "+cu12" in line.lower()]
    results.append(check(
        "requirements.txt does not hard-pin a CPU-only or CUDA-only torch build",
        not bad_pins,
        f"offending lines: {bad_pins}" if bad_pins else "no hardware-specific wheel pins found",
    ))

    if cuda_available:
        try:
            t = torch.randn(4, 4).to("cuda")
            _ = t @ t
            results.append(check("CUDA tensor operation executes", True, torch.cuda.get_device_name(0)))
        except Exception as e:  # noqa: BLE001
            results.append(check("CUDA tensor operation executes", False, str(e)))
    else:
        print("\nNOTE: No CUDA device detected on this machine.")
        print("GPU training has NOT been performed on this machine.")
        print("See docs/gpu_training_guide.md for the exact steps to run on the NVIDIA machine.")

    n_pass = sum(1 for r in results if r["passed"])
    lines = [
        "# GPU Handoff Readiness Check",
        "",
        f"Run on: {dev_report['platform']}",
        f"CUDA available on this machine: **{cuda_available}**",
        "",
        "**GPU training has NOT been performed on this machine.** "
        "The checks below verify device-portability of the code, not that training on a GPU occurred.",
        "",
        f"Result: {n_pass}/{len(results)} checks passed.",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['name']} | {'PASS' if r['passed'] else 'FAIL'} | {r['detail']} |")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{n_pass}/{len(results)} checks passed. Report: {REPORT_PATH}")

    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
