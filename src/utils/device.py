"""
Automatic CPU/CUDA device selection.

Used everywhere a tensor or model needs a device so the SAME code runs
unmodified on this CPU-only development laptop and on the future NVIDIA
GPU machine -- nothing in this project hard-codes "cpu" or "cuda".
"""

from __future__ import annotations

import platform
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report() -> dict:
    """Human-readable snapshot of the hardware actually available right now."""
    report = {
        "selected_device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    if torch.cuda.is_available():
        report["cuda_device_name"] = torch.cuda.get_device_name(0)
        report["cuda_device_count"] = torch.cuda.device_count()
        report["cuda_version"] = torch.version.cuda
    else:
        report["note"] = "GPU training has NOT been performed on this machine (no CUDA device detected)."
    return report
