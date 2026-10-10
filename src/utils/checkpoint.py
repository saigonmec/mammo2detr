"""Checkpoint I/O and top-k bookkeeping."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def extract_model_state(ckpt: dict[str, Any], prefer_ema: bool = True) -> tuple[dict[str, Any], str]:
    if "model" in ckpt or "ema" in ckpt:
        if prefer_ema and ckpt.get("ema"):
            return strip_module_prefix(ckpt["ema"]["module"]), "ema"
        return strip_module_prefix(ckpt["model"]), "model"
    return strip_module_prefix(ckpt), "raw"


def load_model_weights(model: torch.nn.Module, path: str | Path, strict: bool = True, prefer_ema: bool = True) -> None:
    ckpt = load_checkpoint(path)
    state, source = extract_model_state(ckpt, prefer_ema)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"State dict mismatch. Missing: {missing[:10]} Unexpected: {unexpected[:10]}")
    if missing or unexpected:
        logger.warning("Loaded with missing=%d unexpected=%d keys", len(missing), len(unexpected))
    logger.info("Loaded weights from %s (%s)", path, source)


class TopKKeeper:
    """Keep the ``k`` best checkpoints on disk according to a monitored metric."""

    def __init__(self, directory: str | Path, k: int, mode: str = "max"):
        self.directory = Path(directory)
        self.k = k
        self.mode = mode
        self.entries: list[tuple[float, Path]] = []

    def is_better(self, value: float, best: float | None) -> bool:
        if best is None:
            return True
        return value > best if self.mode == "max" else value < best

    def maybe_add(self, value: float, epoch: int, state: dict[str, Any], metric_name: str) -> Path | None:
        if self.k <= 0:
            return None
        safe = metric_name.replace("/", "_")
        path = self.directory / f"epoch{epoch:03d}_{safe}{value:.4f}.pt"
        candidates = self.entries + [(value, path)]
        candidates.sort(key=lambda e: e[0], reverse=(self.mode == "max"))
        keep = candidates[: self.k]
        if (value, path) not in keep:
            return None
        save_checkpoint(path, state)
        for v, p in candidates[self.k :]:
            if p.exists():
                p.unlink()
        self.entries = keep
        return path

    def state_dict(self) -> list[tuple[float, str]]:
        return [(v, str(p)) for v, p in self.entries]

    def load_state_dict(self, entries: list[tuple[float, str]]) -> None:
        self.entries = [(float(v), Path(p)) for v, p in entries if Path(p).exists()]
