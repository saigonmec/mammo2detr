"""Assemble datasets and loaders from a config."""
from __future__ import annotations

import logging

from torch.utils.data import DataLoader

from ..config import Config
from ..utils.misc import worker_init_fn
from .dataset import MammoDetectionDataset, collate_fn
from .metadata import MetadataBundle, load_metadata
from .samplers import build_train_sampler
from .transforms import build_eval_transform, build_train_transform

logger = logging.getLogger(__name__)


def build_dataloaders(cfg: Config, rank: int = 0, world_size: int = 1, splits: tuple[str, ...] = ("train", "val", "test")) -> tuple[MetadataBundle, dict[str, DataLoader]]:
    d = cfg.data
    bundle = load_metadata(d, seed=cfg.seed)
    loaders: dict[str, DataLoader] = {}
    name_map = {"train": d.train_split, "val": d.val_split, "test": d.test_split}
    for key in splits:
        recs = bundle[name_map[key]]
        if not recs:
            continue
        is_train = key == "train"
        ds = MammoDetectionDataset(recs, d, build_train_transform(d) if is_train else build_eval_transform(d))
        sampler = None
        if is_train:
            target = d.balanced_sampler_target or (d.image_labels[0].name if d.image_labels else None)
            sampler = build_train_sampler(recs, target, rank, world_size, cfg.seed, d.balanced_sampler)
        elif world_size > 1:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
        loaders[key] = DataLoader(
            ds,
            batch_size=d.batch_size,
            shuffle=(is_train and sampler is None),
            sampler=sampler,
            num_workers=d.num_workers,
            pin_memory=d.pin_memory,
            drop_last=(is_train and d.drop_last),
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=d.num_workers > 0,
        )
        logger.info("Loader %s: %d images, %d batches", key, len(ds), len(loaders[key]))
    return bundle, loaders
