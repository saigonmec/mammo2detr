"""Lesion detection metrics for mammography.

* COCO-style average precision at configurable IoU thresholds (+ mAP@[.5:.95]).
* FROC: lesion sensitivity vs. false positives per image (negatives included),
  with IoU or center-point matching, summarised at standard FPPI operating points.
* Localisation recall@k on positive images.
* Image-level AUC from the max detection score.

Every function takes a list of per-image dicts::

    {"pred_boxes": [P,4] xyxy, "pred_scores": [P], "pred_labels": [P],
     "gt_boxes": [G,4] xyxy, "gt_labels": [G]}
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .metrics_cls import binary_auc


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def hit_matrix(pred: np.ndarray, gt: np.ndarray, criterion: str, iou_thr: float) -> np.ndarray:
    """[P, G] bool: prediction p "hits" ground truth g."""
    if criterion == "iou":
        return iou_matrix(pred, gt) >= iou_thr
    if criterion == "center":
        pred = np.asarray(pred, dtype=np.float64).reshape(-1, 4)
        gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
        if len(pred) == 0 or len(gt) == 0:
            return np.zeros((len(pred), len(gt)), dtype=bool)
        pc = (pred[:, :2] + pred[:, 2:]) / 2
        gc = (gt[:, :2] + gt[:, 2:]) / 2
        p_in_g = (pc[:, None, 0] >= gt[None, :, 0]) & (pc[:, None, 0] <= gt[None, :, 2]) & (pc[:, None, 1] >= gt[None, :, 1]) & (pc[:, None, 1] <= gt[None, :, 3])
        g_in_p = (gc[None, :, 0] >= pred[:, None, 0]) & (gc[None, :, 0] <= pred[:, None, 2]) & (gc[None, :, 1] >= pred[:, None, 1]) & (gc[None, :, 1] <= pred[:, None, 3])
        return p_in_g | g_in_p
    raise ValueError(f"Unknown match criterion '{criterion}'")


def _filter_class(img: dict[str, Any], class_id: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pb = np.asarray(img["pred_boxes"], dtype=np.float64).reshape(-1, 4)
    ps = np.asarray(img["pred_scores"], dtype=np.float64).reshape(-1)
    gb = np.asarray(img["gt_boxes"], dtype=np.float64).reshape(-1, 4)
    if class_id is not None:
        pl = np.asarray(img.get("pred_labels", np.zeros(len(pb))), dtype=np.int64).reshape(-1)
        gl = np.asarray(img.get("gt_labels", np.zeros(len(gb))), dtype=np.int64).reshape(-1)
        pb, ps, gb = pb[pl == class_id], ps[pl == class_id], gb[gl == class_id]
    return pb, ps, gb


def average_precision_det(images: list[dict[str, Any]], iou_thr: float = 0.5, class_id: int | None = None) -> dict[str, float]:
    """COCO-style AP (101-point interpolation) with greedy per-image matching."""
    scores, matches = [], []
    n_gt = 0
    for img in images:
        pb, ps, gb = _filter_class(img, class_id)
        n_gt += len(gb)
        if len(pb) == 0:
            continue
        order = np.argsort(-ps, kind="mergesort")
        pb, ps = pb[order], ps[order]
        ious = iou_matrix(pb, gb)
        gt_used = np.zeros(len(gb), dtype=bool)
        for p in range(len(pb)):
            tp = False
            if len(gb):
                cand = np.where((~gt_used) & (ious[p] >= iou_thr))[0]
                if len(cand):
                    g = cand[np.argmax(ious[p, cand])]
                    gt_used[g] = True
                    tp = True
            scores.append(ps[p])
            matches.append(tp)
    if n_gt == 0:
        return {"ap": float("nan"), "precision": float("nan"), "recall": float("nan"), "n_gt": 0.0}
    if not scores:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0, "n_gt": float(n_gt)}
    scores_a = np.asarray(scores)
    matches_a = np.asarray(matches, dtype=bool)
    order = np.argsort(-scores_a, kind="mergesort")
    tp = np.cumsum(matches_a[order])
    fp = np.cumsum(~matches_a[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    # precision envelope
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    rec_points = np.linspace(0, 1, 101)
    inds = np.searchsorted(recall, rec_points, side="left")
    q = np.array([precision[i] if i < len(precision) else 0.0 for i in inds])
    return {"ap": float(q.mean()), "precision": float(precision[-1]), "recall": float(recall[-1]), "n_gt": float(n_gt)}


def froc(images: list[dict[str, Any]], fppi_targets: list[float], criterion: str = "iou", iou_thr: float = 0.25, class_id: int | None = None) -> dict[str, Any]:
    """FROC analysis. Returns sensitivities at the requested FPPI operating points,
    their mean, and the full curve ``(fppi, sensitivity, threshold)``."""
    n_images = len(images)
    gt_scores: list[float] = []  # per GT: max score of any prediction hitting it (-inf if none)
    fp_scores: list[float] = []  # scores of predictions hitting no GT
    all_scores: list[float] = []
    for img in images:
        pb, ps, gb = _filter_class(img, class_id)
        all_scores.extend(ps.tolist())
        if len(gb) == 0:
            fp_scores.extend(ps.tolist())
            continue
        hits = hit_matrix(pb, gb, criterion, iou_thr) if len(pb) else np.zeros((0, len(gb)), dtype=bool)
        for g in range(len(gb)):
            s = ps[hits[:, g]]
            gt_scores.append(float(s.max()) if len(s) else -np.inf)
        if len(pb):
            fp_scores.extend(ps[~hits.any(1)].tolist())
    n_gt = len(gt_scores)
    out: dict[str, Any] = {"n_gt": float(n_gt), "n_images": float(n_images)}
    if n_gt == 0 or n_images == 0:
        for t in fppi_targets:
            out[f"sens@fppi{t:g}"] = float("nan")
        out["froc_mean"] = float("nan")
        out["curve"] = []
        return out
    thresholds = np.unique(np.asarray(all_scores, dtype=np.float64)) if all_scores else np.array([])
    thresholds = thresholds[::-1]  # descending
    gt_a = np.asarray(gt_scores)
    fp_sorted = np.sort(np.asarray(fp_scores)) if fp_scores else np.array([])
    curve = []
    for t in thresholds:
        sens = float((gt_a >= t).mean())
        n_fp = len(fp_sorted) - np.searchsorted(fp_sorted, t, side="left") if len(fp_sorted) else 0
        curve.append((float(n_fp) / n_images, sens, float(t)))
    curve.sort(key=lambda c: (c[0], c[1]))
    fppis = np.array([c[0] for c in curve])
    senss = np.array([c[1] for c in curve])
    vals = []
    for target in fppi_targets:
        ok = fppis <= target
        v = float(senss[ok].max()) if ok.any() else 0.0
        out[f"sens@fppi{target:g}"] = v
        vals.append(v)
    out["froc_mean"] = float(np.mean(vals)) if vals else float("nan")
    out["curve"] = curve
    return out


def localization_recall(images: list[dict[str, Any]], k: int, criterion: str, iou_thr: float) -> dict[str, float]:
    """On positive images: fraction of images where any top-k prediction hits a GT,
    and fraction of GT lesions hit by a top-k prediction."""
    n_pos_img = hit_img = n_gt = hit_gt = 0
    for img in images:
        pb, ps, gb = _filter_class(img, None)
        if len(gb) == 0:
            continue
        n_pos_img += 1
        n_gt += len(gb)
        if len(pb) == 0:
            continue
        top = np.argsort(-ps, kind="mergesort")[:k]
        hits = hit_matrix(pb[top], gb, criterion, iou_thr)
        hit_img += int(hits.any())
        hit_gt += int(hits.any(0).sum())
    return {
        f"img_recall@{k}": hit_img / n_pos_img if n_pos_img else float("nan"),
        f"lesion_recall@{k}": hit_gt / n_gt if n_gt else float("nan"),
    }


def detection_summary(images: list[dict[str, Any]], ap_iou_thresholds: list[float], fppi_targets: list[float], froc_match: str, froc_iou: float, topk: int, det_class_names: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for thr in ap_iou_thresholds:
        out[f"ap{int(round(thr * 100))}"] = average_precision_det(images, thr)["ap"]
    coco = [average_precision_det(images, t)["ap"] for t in np.arange(0.5, 0.96, 0.05)]
    out["map_coco"] = float(np.nanmean(coco)) if coco else float("nan")
    fr = froc(images, fppi_targets, froc_match, froc_iou)
    out.update({k: v for k, v in fr.items() if k != "curve"})
    out["froc_curve"] = fr["curve"]
    out.update(localization_recall(images, topk, froc_match, froc_iou))
    out.update(localization_recall(images, 1, froc_match, froc_iou))
    # image-level discrimination from the max detection score (positives = images with GT)
    has_gt = np.array([len(np.asarray(im["gt_boxes"]).reshape(-1, 4)) > 0 for im in images]).astype(int)
    max_score = np.array([float(np.max(im["pred_scores"])) if len(im["pred_scores"]) else 0.0 for im in images])
    out["box_img_auc"] = binary_auc(has_gt, max_score) if len(images) else float("nan")
    out["mean_fp_per_image"] = float(np.mean([int((np.asarray(im["pred_scores"]) >= 0.5).sum()) for im in images])) if images else float("nan")
    if det_class_names and len(det_class_names) > 1:
        for cid, name in enumerate(det_class_names):
            out[f"ap50_{name}"] = average_precision_det(images, 0.5, cid)["ap"]
            out[f"froc_mean_{name}"] = froc(images, fppi_targets, froc_match, froc_iou, cid)["froc_mean"]
    return out
