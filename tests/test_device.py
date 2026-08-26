import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.device import device_report, get_device


def test_get_device_returns_torch_device():
    d = get_device()
    assert isinstance(d, torch.device)
    assert d.type in ("cpu", "cuda")


def test_get_device_matches_cuda_availability():
    d = get_device()
    if torch.cuda.is_available():
        assert d.type == "cuda"
    else:
        assert d.type == "cpu"


def test_device_report_contains_required_fields():
    report = device_report()
    assert "selected_device" in report
    assert "cuda_available" in report
    assert isinstance(report["cuda_available"], bool)
    if not report["cuda_available"]:
        assert "note" in report
        assert "not been performed" in report["note"].lower()
