"""Evaluate a checkpoint on a split of the metadata-described dataset.

    python -m src.evaluate --checkpoint runs/<run>/best.pt --split test --data-root /data/mammo
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .data import build_dataloaders
from .engine.evaluator import KEY_METRICS, export_predictions, format_metrics, run_evaluation
from .engine.postprocess import PostProcess
from .losses import build_criterion
from .models import build_model
from .utils.checkpoint import extract_model_state, load_checkpoint
from .utils.logger import setup_logging
from .utils.misc import resolve_device, seed_everything
from .viz.draw import draw_detections, save_prediction_image


def config_from_checkpoint(ckpt: dict, config_path: str | None, overrides: list[str]) -> Config:
    if config_path is not None:
        return Config.load(config_path, overrides)
    from .config import apply_overrides

    d = apply_overrides(dict(ckpt["config"]), overrides)
    return Config.from_dict(d)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Evaluate MammoDETR")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--config", default=None, help="override the config stored in the checkpoint")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--viz", type=int, default=16, help="number of visualisations to save")
    p.add_argument("--no-ema", action="store_true", help="use raw weights instead of EMA weights")
    args = p.parse_args(argv)

    ckpt = load_checkpoint(args.checkpoint)
    overrides = list(args.overrides)
    if args.data_root:
        overrides.append(f"data.root={args.data_root}")
    if args.device:
        overrides.append(f"device={args.device}")
    if args.batch_size:
        overrides.append(f"data.batch_size={args.batch_size}")
    cfg = config_from_checkpoint(ckpt, args.config, overrides)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / f"eval_{args.split}"
    setup_logging(out_dir)
    log = logging.getLogger("src.evaluate")
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    bundle, loaders = build_dataloaders(cfg, splits=(args.split,))
    if args.split not in loaders:
        raise SystemExit(f"split '{args.split}' has no images")
    model = build_model(cfg)
    state, source = extract_model_state(ckpt, prefer_ema=not args.no_ema)
    model.load_state_dict(state)
    model.to(device)
    log.info("Loaded %s weights from %s", source, args.checkpoint)
    criterion = build_criterion(cfg, bundle.image_class_counts).to(device)
    out = run_evaluation(model, loaders[args.split], device, cfg, criterion, PostProcess(cfg.eval.topk, cfg.eval.score_threshold, cfg.eval.nms_iou), bundle.image_class_names, bundle.det_class_names, amp=cfg.optim.amp and device.type == "cuda", desc=args.split)
    metrics = out.scalar_metrics()
    log.info("%s: %s", args.split, format_metrics(metrics, KEY_METRICS))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out.metrics, f, indent=2, default=float)
    export_predictions(out.per_image, out_dir, bundle.image_class_names, bundle.det_class_names)
    n = 0
    for r in out.per_image:
        if n >= args.viz:
            break
        if not Path(r.get("path", "")).exists():
            continue
        img = cv2.cvtColor(cv2.imread(r["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        title = " ".join(f"{k}={v}" for k, v in r.get("image_labels", {}).items())
        vis = draw_detections(img, r["gt_boxes"], r["pred_boxes"], r["pred_scores"], r["pred_labels"], bundle.det_class_names, 0.3, title)
        save_prediction_image(out_dir / "viz" / f"{r['image_id']}.jpg", vis)
        n += 1
    log.info("Saved metrics, predictions and %d visualisations to %s", n, out_dir)
    return metrics


if __name__ == "__main__":
    main()
