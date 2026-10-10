"""Training loop with AMP, gradient clipping/accumulation, EMA, schedulers,
checkpointing (last / best / top-k), resume, early stopping and evaluation."""
from __future__ import annotations

import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import Config
from ..data.metadata import MetadataBundle
from ..utils.checkpoint import TopKKeeper, load_checkpoint, load_model_weights, save_checkpoint, strip_module_prefix
from ..utils.ema import ModelEMA
from ..utils.logger import ExperimentLogger
from ..utils.misc import DistInfo, MetricTracker, Timer, count_parameters, format_seconds, is_main_process, to_device, unwrap_model
from ..utils.scheduler import Scheduler, build_optimizer
from ..viz.draw import draw_detections, save_prediction_image
from .evaluator import KEY_METRICS, EvalOutput, export_predictions, format_metrics, run_evaluation
from .postprocess import PostProcess

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        criterion: nn.Module,
        loaders: dict[str, Any],
        bundle: MetadataBundle,
        device: torch.device,
        dist_info: DistInfo | None = None,
        output_dir: str | Path = "runs/debug",
    ):
        self.cfg = cfg
        self.device = device
        self.dist = dist_info or DistInfo()
        self.output_dir = Path(output_dir)
        self.loaders = loaders
        self.bundle = bundle
        self.criterion = criterion.to(device)
        self.raw_model = model.to(device)
        if self.dist.distributed:
            self.model: nn.Module = DDP(self.raw_model, device_ids=[self.dist.local_rank] if device.type == "cuda" else None, find_unused_parameters=True)
        else:
            self.model = self.raw_model

        o = cfg.optim
        self.accum = max(1, o.accumulate_steps)
        self.steps_per_epoch = max(1, len(loaders["train"]) // self.accum)
        self.optimizer = build_optimizer(self.raw_model, o)
        self.scheduler = Scheduler(self.optimizer, o, self.steps_per_epoch)
        self.amp = bool(o.amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.ema = None
        if o.ema.enabled:
            total_steps = max(1, self.steps_per_epoch * o.epochs)
            # never let the EMA averaging window exceed half of the run (short runs would otherwise
            # be evaluated with weights that lag far behind the trained model)
            decay = min(o.ema.decay, 1.0 - 2.0 / total_steps)
            if decay < o.ema.decay:
                logger.warning("EMA decay lowered from %.5f to %.5f because the run has only %d steps", o.ema.decay, decay, total_steps)
            self.ema = ModelEMA(self.raw_model, decay, min(o.ema.warmup_steps, max(1, total_steps // 10)))
        self.postprocess = PostProcess(cfg.eval.topk, cfg.eval.score_threshold, cfg.eval.nms_iou)

        self.global_step = 0
        self.start_epoch = 0
        self.best_value: float | None = None
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.monitor = cfg.eval.monitor
        self.monitor_mode = cfg.eval.monitor_mode
        self._monitor_warned = False
        self.keeper = TopKKeeper(self.output_dir / "checkpoints", cfg.logging.save_top_k, self.monitor_mode)
        self.exp_logger: ExperimentLogger | None = None
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            cfg.save(self.output_dir / "config.yaml")
            self.exp_logger = ExperimentLogger(self.output_dir, cfg.logging.tensorboard, cfg.logging.wandb, cfg.logging.wandb_project, cfg.logging.run_name, cfg.to_dict())
        if cfg.resume:
            self.load(cfg.resume)
        elif cfg.init_weights:
            load_model_weights(self.raw_model, cfg.init_weights, strict=False)
            if self.ema is not None:
                self.ema.module.load_state_dict(self.raw_model.state_dict())
        logger.info(
            "Trainer: device=%s amp=%s ema=%s params=%.2fM (trainable %.2fM) steps/epoch=%d",
            device, self.amp, self.ema is not None, count_parameters(self.raw_model) / 1e6, count_parameters(self.raw_model, True) / 1e6, self.steps_per_epoch,
        )

    # ------------------------------------------------------------------ #
    def state(self, epoch: int) -> dict[str, Any]:
        return {
            "model": self.raw_model.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "topk": self.keeper.state_dict(),
            "config": self.cfg.to_dict(),
            "image_class_names": self.bundle.image_class_names,
            "det_class_names": self.bundle.det_class_names,
        }

    def load(self, path: str | Path) -> None:
        ckpt = load_checkpoint(path, map_location=self.device)
        self.raw_model.load_state_dict(strip_module_prefix(ckpt["model"]))
        if self.ema is not None and ckpt.get("ema"):
            self.ema.load_state_dict(ckpt["ema"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt["scaler"]:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_value = ckpt.get("best_value")
        self.best_epoch = int(ckpt.get("best_epoch", -1))
        self.keeper.load_state_dict(ckpt.get("topk", []))
        logger.info("Resumed from %s at epoch %d (best %s=%s @ %d)", path, self.start_epoch, self.monitor, self.best_value, self.best_epoch)

    # ------------------------------------------------------------------ #
    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        loader = self.loaders["train"]
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        tracker = MetricTracker()
        timer = Timer()
        n_iters = len(loader)
        self.optimizer.zero_grad(set_to_none=True)
        log_every = max(1, self.cfg.logging.log_every)
        for it, (images, masks, targets) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            targets = to_device(targets, self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                outputs = self.model(images, masks, targets)
                losses = self.criterion(outputs, targets)
                loss = losses["loss_total"] / self.accum
            if not torch.isfinite(loss):
                logger.warning("Non-finite loss at epoch %d iter %d: %s - skipping batch", epoch, it, {k: float(v) for k, v in losses.items() if v.ndim == 0})
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.scaler.scale(loss).backward()
            if (it + 1) % self.accum == 0 or (it + 1) == n_iters:
                self.scaler.unscale_(self.optimizer)
                if self.cfg.optim.grad_clip and self.cfg.optim.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.cfg.optim.grad_clip)
                    tracker.update(grad_norm=float(grad_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step_iter()
                if self.ema is not None:
                    self.ema.update(self.raw_model)
                self.global_step += 1
            tracker.update(n=images.shape[0], **{k: float(v) for k, v in losses.items() if v.ndim == 0 and ("aux" not in k)})
            if is_main_process() and ((it + 1) % log_every == 0 or (it + 1) == n_iters):
                el = timer.elapsed()
                eta = el / (it + 1) * (n_iters - it - 1)
                mem = f" mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G" if self.device.type == "cuda" else ""
                m = tracker.meters
                logger.info(
                    "ep %d [%d/%d] loss=%.3f ce=%.3f bbox=%.3f giou=%.3f%s lr=%.2e %.2fit/s eta=%s%s",
                    epoch, it + 1, n_iters, m["loss_total"].avg, m["loss_ce"].avg, m["loss_bbox"].avg, m["loss_giou"].avg,
                    "".join(f" {k[9:]}=%.3f" % v.avg for k, v in m.items() if k.startswith("loss_img_")),
                    self.scheduler.lr, (it + 1) / max(el, 1e-6), format_seconds(eta), mem,
                )
                if self.exp_logger is not None:
                    self.exp_logger.log_scalars({k: v.avg for k, v in m.items()}, self.global_step, prefix="train_iter/")
        avgs = tracker.global_avgs()
        avgs["epoch_time"] = timer.elapsed()
        return avgs

    # ------------------------------------------------------------------ #
    def eval_model(self) -> nn.Module:
        return self.ema.module if self.ema is not None else self.raw_model

    def evaluate(self, split: str, desc: str | None = None) -> EvalOutput:
        loader = self.loaders.get(split)
        if loader is None:
            return EvalOutput()
        return run_evaluation(
            self.eval_model(), loader, self.device, self.cfg, self.criterion, self.postprocess,
            self.bundle.image_class_names, self.bundle.det_class_names, amp=self.amp, desc=desc or split,
        )

    def _monitor_value(self, metrics: dict[str, Any]) -> tuple[float | None, str, str]:
        v = metrics.get(self.monitor)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v), self.monitor, self.monitor_mode
        if not self._monitor_warned:
            logger.warning("Monitor metric '%s' unavailable (no boxes in val split?). Falling back to val loss (min).", self.monitor)
            self._monitor_warned = True
        loss = metrics.get("loss")
        if isinstance(loss, (int, float)) and not math.isnan(loss):
            return float(loss), "loss", "min"
        return None, self.monitor, self.monitor_mode

    def save_visualizations(self, out: EvalOutput, split: str, epoch: int) -> None:
        n = self.cfg.eval.viz_samples
        if n <= 0 or not is_main_process() or not out.per_image:
            return
        pos = [r for r in out.per_image if len(np.asarray(r["gt_boxes"]).reshape(-1, 4)) > 0]
        neg = [r for r in out.per_image if len(np.asarray(r["gt_boxes"]).reshape(-1, 4)) == 0]
        chosen = pos[: max(1, n * 2 // 3)] + neg[: n - min(len(pos), max(1, n * 2 // 3))]
        vdir = self.output_dir / "viz" / f"{split}_epoch{epoch:03d}"
        for r in chosen[:n]:
            if not r.get("path") or not Path(r["path"]).exists():
                continue
            img = cv2.cvtColor(cv2.imread(r["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            title = " ".join(f"{k}={v}" for k, v in r.get("image_labels", {}).items())
            if "image_probs" in r:
                title += " | " + " ".join(f"p({k})={p[-1] if len(p) == 2 else p.max():.2f}" for k, p in r["image_probs"].items())
            vis = draw_detections(img, r["gt_boxes"], r["pred_boxes"], r["pred_scores"], r["pred_labels"], self.bundle.det_class_names, 0.3, title)
            save_prediction_image(vdir / f"{r['image_id']}.jpg", vis)

    # ------------------------------------------------------------------ #
    def fit(self) -> dict[str, Any]:
        cfg = self.cfg
        epochs = cfg.optim.epochs
        val_split = "val" if "val" in self.loaders else ("test" if "test" in self.loaders else None)
        if val_split == "test":
            logger.warning("No validation split: model selection will use the TEST split (optimistic estimates).")
        total_timer = Timer()
        stop = False
        epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, epochs):
            train_metrics = self.train_one_epoch(epoch)
            row: dict[str, Any] = {"epoch": epoch, "step": self.global_step, "lr": self.scheduler.lr}
            row.update({f"train/{k}": v for k, v in train_metrics.items()})
            do_eval = val_split is not None and ((epoch + 1) % cfg.eval.eval_every == 0 or epoch + 1 == epochs)
            value: float | None = None
            if do_eval:
                out = self.evaluate(val_split)
                metrics = out.scalar_metrics()
                row.update({f"{val_split}/{k}": v for k, v in metrics.items()})
                self.scheduler.step_epoch(metrics.get("loss"))
                value, mon_name, mon_mode = self._monitor_value(metrics)
                if is_main_process():
                    logger.info("ep %d %s: %s", epoch, val_split, format_metrics(metrics, KEY_METRICS))
                    self.save_visualizations(out, val_split, epoch)
            improved = False
            if value is not None:
                improved = self.keeper.is_better(value, self.best_value) if mon_mode == self.monitor_mode else (value < (self.best_value if self.best_value is not None else float("inf")))
                if improved:
                    self.best_value, self.best_epoch = value, epoch
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
            row["best_value"] = self.best_value
            row["best_epoch"] = self.best_epoch
            if is_main_process():
                logger.info(
                    "ep %d done in %s | train loss %.4f | %s=%s (best %s @ ep %d) | elapsed %s",
                    epoch, format_seconds(train_metrics["epoch_time"]), train_metrics.get("loss_total", float("nan")), self.monitor,
                    f"{value:.4f}" if value is not None else "n/a", f"{self.best_value:.4f}" if self.best_value is not None else "n/a", self.best_epoch, total_timer,
                )
                if self.exp_logger is not None:
                    self.exp_logger.log_epoch(row)
                    self.exp_logger.log_scalars({k: v for k, v in row.items() if isinstance(v, (int, float))}, epoch, prefix="epoch/")
                state = self.state(epoch)
                if cfg.logging.save_last:
                    save_checkpoint(self.output_dir / "last.pt", state)
                if improved:
                    save_checkpoint(self.output_dir / "best.pt", state)
                if value is not None:
                    self.keeper.maybe_add(value, epoch, state, self.monitor)
            stop = False
            pat = cfg.optim.early_stopping_patience
            if pat is not None and self.epochs_without_improvement >= pat:
                logger.info("Early stopping: no improvement for %d epochs", pat)
                stop = True
            if self.dist.distributed:
                t = torch.tensor([int(stop)], device=self.device)
                dist.broadcast(t, src=0)
                stop = bool(t.item())
            if stop:
                break
        summary: dict[str, Any] = {"best_value": self.best_value, "best_epoch": self.best_epoch, "monitor": self.monitor, "epochs_run": (epoch + 1) if stop else epochs}
        if cfg.eval.eval_on_test_at_end and "test" in self.loaders:
            summary["test"] = self.final_test()
        if self.exp_logger is not None:
            self.exp_logger.close()
        if is_main_process():
            with open(self.output_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=float)
        return summary

    def final_test(self) -> dict[str, Any]:
        best = self.output_dir / "best.pt"
        if best.exists():
            ckpt = load_checkpoint(best, map_location=self.device)
            self.raw_model.load_state_dict(strip_module_prefix(ckpt["model"]))
            if self.ema is not None and ckpt.get("ema"):
                self.ema.load_state_dict(ckpt["ema"])
            logger.info("Loaded best checkpoint (epoch %s) for final test evaluation", ckpt.get("epoch"))
        out = self.evaluate("test", desc="test")
        metrics = out.scalar_metrics()
        if is_main_process():
            logger.info("TEST: %s", format_metrics(metrics, KEY_METRICS))
            tdir = self.output_dir / "test"
            tdir.mkdir(parents=True, exist_ok=True)
            with open(tdir / "metrics.json", "w") as f:
                json.dump({k: v for k, v in out.metrics.items()}, f, indent=2, default=float)
            export_predictions(out.per_image, tdir, self.bundle.image_class_names, self.bundle.det_class_names)
            self.save_visualizations(out, "test", self.best_epoch if self.best_epoch >= 0 else 0)
        return metrics
