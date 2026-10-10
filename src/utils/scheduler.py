"""Optimizer with parameter groups and learning-rate schedules
(per-iteration warmup + cosine/step/multistep, or epoch-level plateau)."""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from ..config import OptimConfig


class Scheduler:
    """Thin wrapper exposing ``step_iter()`` (every optimizer step) and ``step_epoch(metric)``."""

    def __init__(self, optimizer: Optimizer, cfg: OptimConfig, steps_per_epoch: int):
        self.cfg = cfg
        self.optimizer = optimizer
        self.steps_per_epoch = max(1, steps_per_epoch)
        total = cfg.epochs * self.steps_per_epoch
        warmup = int(cfg.warmup_epochs * self.steps_per_epoch)
        self.plateau: ReduceLROnPlateau | None = None
        name = cfg.scheduler

        def warm(s: int) -> float:
            return min(1.0, (s + 1) / warmup) if warmup > 0 else 1.0

        if name == "plateau":
            self.plateau = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=cfg.plateau_patience, min_lr=cfg.lr * cfg.min_lr_ratio)
            fn = warm
        elif name == "cosine":

            def fn(s: int) -> float:
                if warmup > 0 and s < warmup:
                    return (s + 1) / warmup
                p = min(1.0, (s - warmup) / max(1, total - warmup))
                return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

        elif name in ("step", "multistep"):
            milestones = sorted(m * self.steps_per_epoch for m in cfg.step_milestones)

            def fn(s: int) -> float:
                if warmup > 0 and s < warmup:
                    return (s + 1) / warmup
                return cfg.step_gamma ** sum(s >= m for m in milestones)

        elif name == "none":
            fn = warm
        else:
            raise ValueError(f"Unknown scheduler '{name}'")
        self.iter_sched = LambdaLR(optimizer, fn)

    def step_iter(self) -> None:
        self.iter_sched.step()

    def step_epoch(self, metric: float | None = None) -> None:
        if self.plateau is not None and metric is not None:
            self.plateau.step(metric)

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> dict:
        return {"iter": self.iter_sched.state_dict(), "plateau": self.plateau.state_dict() if self.plateau is not None else None}

    def load_state_dict(self, state: dict) -> None:
        if state.get("iter") is not None:
            self.iter_sched.load_state_dict(state["iter"])
        if self.plateau is not None and state.get("plateau") is not None:
            self.plateau.load_state_dict(state["plateau"])


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """AdamW/SGD with (backbone lr multiplier) x (no weight decay for norms/biases/embeddings)."""
    groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        no_decay = p.ndim <= 1 or name.endswith(".bias") or "embed" in name
        groups.setdefault((is_backbone, no_decay), []).append(p)
    param_groups = []
    for (is_backbone, no_decay), params in sorted(groups.items()):
        param_groups.append(
            {
                "params": params,
                "lr": cfg.lr * (cfg.backbone_lr_mult if is_backbone else 1.0),
                "weight_decay": 0.0 if no_decay else cfg.weight_decay,
                "name": f"{'backbone' if is_backbone else 'head'}{'_nodecay' if no_decay else ''}",
            }
        )
    if cfg.name == "adamw":
        return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=tuple(cfg.betas))
    if cfg.name == "sgd":
        return torch.optim.SGD(param_groups, lr=cfg.lr, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer '{cfg.name}'")
