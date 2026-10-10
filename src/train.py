"""Train a MammoDETR model.

Examples::

    python -m src.train --config src/configs/resnet50_448.yaml --set data.root=/data/mammo
    python -m src.train --config src/configs/convnextv2_tiny_640.yaml --set optim.epochs=60 --run-name cnx_640
    torchrun --nproc_per_node=2 -m src.train --config ... (multi-GPU)
"""
from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import torch

from .config import Config
from .data import build_dataloaders
from .engine import Trainer
from .losses import build_criterion
from .models import build_model
from .utils.logger import setup_logging
from .utils.misc import init_distributed, is_main_process, resolve_device, seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MammoDETR")
    p.add_argument("--config", type=str, default=None, help="YAML config (defaults are used when omitted)")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="dotted override, e.g. optim.lr=2e-4")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init-weights", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args(argv)


def load_config_from_args(args: argparse.Namespace) -> Config:
    overrides = list(args.overrides)
    if args.data_root:
        overrides.append(f"data.root={args.data_root}")
    if args.output_dir:
        overrides.append(f"logging.output_dir={args.output_dir}")
    if args.run_name:
        overrides.append(f"logging.run_name={args.run_name}")
    if args.resume:
        overrides.append(f"resume={args.resume}")
    if args.init_weights:
        overrides.append(f"init_weights={args.init_weights}")
    if args.device:
        overrides.append(f"device={args.device}")
    if args.epochs:
        overrides.append(f"optim.epochs={args.epochs}")
    return Config.load(args.config, overrides)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    cfg = load_config_from_args(args)
    dist_info = init_distributed()
    if cfg.logging.run_name is None:
        h, w = cfg.data.img_size
        backbone = cfg.model.backbone.name.replace("timm:", "").replace("/", "_").replace(":", "_")
        cfg.logging.run_name = f"{backbone}_{h}x{w}_{cfg.model.transformer.type}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    output_dir = Path(cfg.logging.output_dir) / cfg.logging.run_name
    setup_logging(output_dir, dist_info.rank)
    log = logging.getLogger("src.train")
    seed_everything(cfg.seed + dist_info.rank, cfg.deterministic)
    device = resolve_device(cfg.device)
    if dist_info.distributed and device.type == "cuda":
        device = torch.device("cuda", dist_info.local_rank)
    if is_main_process():
        log.info("Output dir: %s", output_dir)
        log.info("Config:\n%s", cfg.to_dict())
    bundle, loaders = build_dataloaders(cfg, dist_info.rank, dist_info.world_size)
    model = build_model(cfg)
    criterion = build_criterion(cfg, bundle.image_class_counts)
    trainer = Trainer(cfg, model, criterion, loaders, bundle, device, dist_info, output_dir)
    summary = trainer.fit()
    if is_main_process():
        log.info("Done. best %s=%s at epoch %s", summary["monitor"], summary["best_value"], summary["best_epoch"])
    if dist_info.distributed:
        torch.distributed.destroy_process_group()
    return summary


if __name__ == "__main__":
    main()
