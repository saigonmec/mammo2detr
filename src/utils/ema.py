"""Exponential moving average of model weights (helps DETR-style training)."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup_steps: int = 500):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = max(1, warmup_steps)
        self.updates = 0

    def _decay(self) -> float:
        # ramp the decay up from 0 so that early EMA weights track the model quickly
        return self.decay * (1 - math.exp(-self.updates / self.warmup_steps))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self._decay()
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = int(state.get("updates", 0))
