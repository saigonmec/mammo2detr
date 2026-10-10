"""Samplers: class-balanced sampling that also works under DDP."""
from __future__ import annotations

from collections import Counter
from typing import Iterator

import torch
from torch.utils.data import Sampler

from .metadata import ImageRecord


class DistributedWeightedRandomSampler(Sampler[int]):
    """Weighted sampling with replacement, sharded across ranks, epoch-seeded."""

    def __init__(self, weights: torch.Tensor, num_samples: int, rank: int = 0, world_size: int = 1, seed: int = 0):
        self.weights = weights.double()
        self.num_samples = num_samples
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g)
        return iter(idx[self.rank :: self.world_size].tolist())

    def __len__(self) -> int:
        return (self.num_samples - self.rank + self.world_size - 1) // self.world_size


def build_train_sampler(records: list[ImageRecord], target: str | None, rank: int, world_size: int, seed: int, balanced: bool):
    if balanced and target is not None:
        counts = Counter(r.image_labels[target] for r in records)
        weights = torch.tensor([1.0 / counts[r.image_labels[target]] for r in records])
        return DistributedWeightedRandomSampler(weights, len(records), rank, world_size, seed)
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        class _ListDS(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(records)

        return DistributedSampler(_ListDS(), num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    return None  # plain shuffle
