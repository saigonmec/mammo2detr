"""Evaluation loop: collects per-image predictions and computes all metrics."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..config import Config
from ..utils.box_ops import box_cxcywh_to_xyxy
from ..utils.misc import all_gather_objects, is_main_process, to_device
from .metrics_cls import classification_metrics
from .metrics_det import detection_summary
from .postprocess import PostProcess

logger = logging.getLogger(__name__)


@dataclass
class EvalOutput:
    metrics: dict[str, Any] = field(default_factory=dict)
    per_image: list[dict[str, Any]] = field(default_factory=list)
    loss: float = float("nan")

    def scalar_metrics(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _targets_to_numpy(t: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    vh, vw = t["size"].tolist()
    oh, ow = t["orig_size"].tolist()
    boxes = t["boxes"]
    if len(boxes):
        xyxy = box_cxcywh_to_xyxy(boxes.cpu()) * torch.tensor([ow, oh, ow, oh])
        return xyxy.numpy(), t["labels"].cpu().numpy()
    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: Config,
    criterion: nn.Module | None = None,
    postprocess: PostProcess | None = None,
    image_class_names: dict[str, list[str]] | None = None,
    det_class_names: list[str] | None = None,
    amp: bool = False,
    desc: str = "eval",
    progress: bool = True,
) -> EvalOutput:
    model.eval()
    postprocess = postprocess or PostProcess(cfg.eval.topk, cfg.eval.score_threshold, cfg.eval.nms_iou)
    per_image: list[dict[str, Any]] = []
    cls_true: dict[str, list[int]] = {}
    cls_prob: dict[str, list[np.ndarray]] = {}
    loss_sum, n_batches = 0.0, 0
    it = tqdm(loader, desc=desc, leave=False, disable=not (progress and is_main_process()))
    for images, masks, targets in it:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets_dev = to_device(targets, device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(images, masks)
        if criterion is not None:
            losses = criterion(outputs, targets_dev)
            loss_sum += float(losses["loss_total"])
            n_batches += 1
        results = postprocess(outputs, targets_dev)
        for i, (t, r) in enumerate(zip(targets, results)):
            gt_boxes, gt_labels = _targets_to_numpy(t)
            rec: dict[str, Any] = {
                "image_id": t["image_id"],
                "path": t.get("path", ""),
                "pred_boxes": r["boxes"].float().cpu().numpy(),
                "pred_scores": r["scores"].float().cpu().numpy(),
                "pred_labels": r["labels"].cpu().numpy(),
                "gt_boxes": gt_boxes,
                "gt_labels": gt_labels,
                "image_labels": {k: int(v) for k, v in t["image_labels"].items()},
            }
            if "image_logits" in outputs:
                rec["image_probs"] = {k: torch.softmax(v[i].float(), -1).cpu().numpy() for k, v in outputs["image_logits"].items()}
                for k, p in rec["image_probs"].items():
                    cls_true.setdefault(k, []).append(int(t["image_labels"][k]))
                    cls_prob.setdefault(k, []).append(p)
            per_image.append(rec)

    # gather across ranks
    gathered = all_gather_objects(per_image)
    per_image = [r for part in gathered for r in part]
    # de-duplicate (DistributedSampler may pad)
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in per_image:
        if r["image_id"] in seen:
            continue
        seen.add(r["image_id"])
        unique.append(r)
    per_image = unique
    loss_parts = all_gather_objects((loss_sum, n_batches))
    tot_loss = sum(p[0] for p in loss_parts)
    tot_n = sum(p[1] for p in loss_parts)

    out = EvalOutput(per_image=per_image, loss=(tot_loss / tot_n) if tot_n else float("nan"))
    metrics: dict[str, Any] = {}
    if criterion is not None:
        metrics["loss"] = out.loss
    # image-level classification
    targets_names = list(cfg.data.image_labels)
    for it_cfg in targets_names:
        name = it_cfg.name
        ys = [r["image_labels"][name] for r in per_image if "image_probs" in r]
        ps = [r["image_probs"][name] for r in per_image if "image_probs" in r]
        if not ys:
            continue
        m = classification_metrics(np.asarray(ys), np.stack(ps), it_cfg.positive_index, (image_class_names or {}).get(name))
        for k, v in m.items():
            metrics[f"cls/{name}/{k}"] = v
    # detection
    e = cfg.eval
    det = detection_summary(per_image, e.ap_iou_thresholds, e.froc_fppi, e.froc_match, e.froc_iou, e.topk, det_class_names)
    for k, v in det.items():
        metrics[f"det/{k}"] = v
    out.metrics = metrics
    return out


def format_metrics(metrics: dict[str, Any], keys: list[str] | None = None) -> str:
    items = []
    for k, v in metrics.items():
        if keys is not None and k not in keys:
            continue
        if isinstance(v, float):
            items.append(f"{k}={v:.4f}")
        elif isinstance(v, int):
            items.append(f"{k}={v}")
    return " | ".join(items)


def export_predictions(per_image: list[dict[str, Any]], out_dir: str, image_class_names: dict[str, list[str]] | None = None, det_class_names: list[str] | None = None) -> None:
    """Write ``predictions_images.csv`` (one row per image) and ``predictions_boxes.csv`` (one row per box)."""
    import csv
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_rows, box_rows = [], []
    for r in per_image:
        row: dict[str, Any] = {"image_id": r["image_id"], "path": r.get("path", "")}
        for k, v in r.get("image_labels", {}).items():
            row[f"label_{k}"] = v
        for k, p in r.get("image_probs", {}).items():
            names = (image_class_names or {}).get(k) or [str(i) for i in range(len(p))]
            for i, pv in enumerate(p):
                row[f"prob_{k}_{names[i] if i < len(names) else i}"] = float(pv)
            row[f"pred_{k}"] = int(p.argmax())
        scores = np.asarray(r["pred_scores"])
        row["n_gt_boxes"] = int(len(np.asarray(r["gt_boxes"]).reshape(-1, 4)))
        row["max_box_score"] = float(scores.max()) if len(scores) else 0.0
        row["n_boxes_ge_0.5"] = int((scores >= 0.5).sum())
        img_rows.append(row)
        for b, s, l in zip(np.asarray(r["pred_boxes"]).reshape(-1, 4), scores, np.asarray(r["pred_labels"]).reshape(-1)):
            box_rows.append({"image_id": r["image_id"], "x0": float(b[0]), "y0": float(b[1]), "x1": float(b[2]), "y1": float(b[3]), "score": float(s),
                             "label": int(l), "class": (det_class_names[int(l)] if det_class_names and int(l) < len(det_class_names) else str(int(l)))})
    for name, rows in (("predictions_images.csv", img_rows), ("predictions_boxes.csv", box_rows)):
        if not rows:
            continue
        fields = sorted({k for row in rows for k in row}, key=lambda k: (k != "image_id", k))
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


KEY_METRICS = [
    "loss",
    "cls/cancer/auc",
    "cls/cancer/accuracy",
    "cls/cancer/sensitivity",
    "cls/cancer/specificity",
    "det/ap25",
    "det/ap50",
    "det/froc_mean",
    "det/sens@fppi0.5",
    "det/sens@fppi1",
    "det/sens@fppi2",
    "det/img_recall@1",
    "det/box_img_auc",
]
