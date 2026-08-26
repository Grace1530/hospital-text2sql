"""
Learning-rate schedule: linear warmup followed by inverse-square-root
decay (the schedule from "Attention Is All You Need", implemented directly
rather than imported from a library).

    lr(step) = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
"""

from __future__ import annotations

import torch


class WarmupInverseSqrtLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = max(1, warmup_steps)

        def lr_lambda(step: int) -> float:
            step = max(step, 1)
            scale = d_model ** -0.5
            return scale * min(step ** -0.5, step * self.warmup_steps ** -1.5)

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)
