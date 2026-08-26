import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.scheduler import WarmupInverseSqrtLR


def test_lr_increases_during_warmup_then_decreases():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    sched = WarmupInverseSqrtLR(opt, d_model=64, warmup_steps=100)

    lrs = []
    for _ in range(300):
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    warmup_end = lrs[99]
    assert lrs[0] < warmup_end  # increasing during warmup
    assert lrs[-1] < warmup_end  # decreasing after warmup
    assert all(l > 0 for l in lrs)


def test_scheduler_state_dict_roundtrip():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    sched = WarmupInverseSqrtLR(opt, d_model=64, warmup_steps=50)
    for _ in range(20):
        opt.step()
        sched.step()
    state = sched.state_dict()

    model2 = torch.nn.Linear(4, 4)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1.0)
    sched2 = WarmupInverseSqrtLR(opt2, d_model=64, warmup_steps=50)
    sched2.load_state_dict(state)
    assert sched2.last_epoch == sched.last_epoch
