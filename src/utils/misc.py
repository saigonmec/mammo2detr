"""Small shared helpers: seeding, devices, distributed, meters, padding."""
from __future__ import annotations

import datetime
import logging
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reproducibility / devices
# --------------------------------------------------------------------------- #
def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(preference: str = "auto") -> torch.device:
    """``auto`` -> cuda > mps > cpu. Explicit strings are passed through."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(obj: Any, device: torch.device, non_blocking: bool = False) -> Any:
    """Recursively move tensors inside dict/list/tuple to ``device``."""
    if isinstance(obj, Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device, non_blocking) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device, non_blocking) for v in obj)
    return obj


def worker_init_fn(worker_id: int) -> None:
    """Make numpy/random inside DataLoader workers depend on the torch seed."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------- #
# Distributed (torchrun) helpers
# --------------------------------------------------------------------------- #
@dataclass
class DistInfo:
    distributed: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0


def init_distributed(backend: str | None = None) -> DistInfo:
    """Initialise ``torch.distributed`` from ``torchrun`` env vars when present."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistInfo()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=60))
    dist.barrier()
    return DistInfo(True, rank, world_size, local_rank)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def reduce_tensor(t: Tensor, average: bool = True) -> Tensor:
    if not is_dist():
        return t
    t = t.clone()
    dist.all_reduce(t)
    if average:
        t /= get_world_size()
    return t


def all_gather_objects(obj: Any) -> list[Any]:
    if not is_dist():
        return [obj]
    out: list[Any] = [None] * get_world_size()
    dist.all_gather_object(out, obj)
    return out


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


# --------------------------------------------------------------------------- #
# Meters
# --------------------------------------------------------------------------- #
class SmoothedValue:
    """Track a series of values with a window median/average and a global average."""

    def __init__(self, window_size: int = 20):
        self.deque: deque[float] = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(float(value))
        self.count += n
        self.total += float(value) * n

    @property
    def median(self) -> float:
        return float(np.median(self.deque)) if self.deque else 0.0

    @property
    def avg(self) -> float:
        return float(np.mean(self.deque)) if self.deque else 0.0

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)

    @property
    def value(self) -> float:
        return self.deque[-1] if self.deque else 0.0


class MetricTracker:
    def __init__(self, window_size: int = 20):
        self.meters: dict[str, SmoothedValue] = defaultdict(lambda: SmoothedValue(window_size))

    def update(self, n: int = 1, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                v = v.item()
            self.meters[k].update(v, n)

    def global_avgs(self) -> dict[str, float]:
        return {k: m.global_avg for k, m in self.meters.items()}

    def __str__(self) -> str:
        return " ".join(f"{k}={m.avg:.4f}" for k, m in self.meters.items())


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def __str__(self) -> str:
        return format_seconds(self.elapsed())


def format_seconds(s: float) -> str:
    return str(datetime.timedelta(seconds=int(s)))


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


# --------------------------------------------------------------------------- #
# Padding of variable-size images (NestedTensor-lite)
# --------------------------------------------------------------------------- #
def pad_images(images: Iterable[Tensor], size_divisible: int = 32) -> tuple[Tensor, Tensor]:
    """Pad a list of ``[C, H, W]`` tensors to a common ``[B, C, Hmax, Wmax]`` batch.

    Returns the batch and a boolean mask ``[B, Hmax, Wmax]`` where ``True``
    marks padded pixels (DETR convention).
    """
    images = list(images)
    c = images[0].shape[0]
    h = max(im.shape[1] for im in images)
    w = max(im.shape[2] for im in images)
    if size_divisible > 1:
        h = int(np.ceil(h / size_divisible) * size_divisible)
        w = int(np.ceil(w / size_divisible) * size_divisible)
    batch = images[0].new_zeros((len(images), c, h, w))
    mask = torch.ones((len(images), h, w), dtype=torch.bool)
    for i, im in enumerate(images):
        batch[i, :, : im.shape[1], : im.shape[2]].copy_(im)
        mask[i, : im.shape[1], : im.shape[2]] = False
    return batch, mask


def downsample_mask(mask: Tensor, size: tuple[int, int]) -> Tensor:
    """Nearest-neighbour downsample of a bool padding mask to a feature-map size."""
    return torch.nn.functional.interpolate(mask[None].float(), size=size, mode="nearest")[0].to(torch.bool)
