"""
Standard DETR metrics following COCO evaluation protocol
- mAP@IoU thresholds (0.5, 0.25, etc.)
- Precision, Recall, F1
- IoU calculation utilities
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def box_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes in (cx, cy, w, h) format.

    Args:
        boxes1: (N, 4) tensor
        boxes2: (M, 4) tensor

    Returns:
        iou: (N, M) tensor
    """
    # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
    boxes1_xyxy = torch.zeros_like(boxes1)
    boxes1_xyxy[:, 0] = boxes1[:, 0] - boxes1[:, 2] / 2  # x1
    boxes1_xyxy[:, 1] = boxes1[:, 1] - boxes1[:, 3] / 2  # y1
    boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2] / 2  # x2
    boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3] / 2  # y2

    boxes2_xyxy = torch.zeros_like(boxes2)
    boxes2_xyxy[:, 0] = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_xyxy[:, 1] = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3] / 2

    # Compute intersection
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Compute areas
    area1 = boxes1[:, 2] * boxes1[:, 3]  # (N,)
    area2 = boxes2[:, 2] * boxes2[:, 3]  # (M,)

    # Compute union
    union = area1[:, None] + area2 - inter

    # Compute IoU
    iou = inter / (union + 1e-6)

    return iou


# Fixed compute_ap_at_iou (main bug fix)
def compute_ap_at_iou(
    pred_boxes: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    all_scores = []
    all_matches = []
    total_gt = 0
    for pred_box, gt_box, scores in zip(pred_boxes, gt_boxes, pred_scores):
        if len(pred_box) == 0 or len(gt_box) == 0:
            all_scores.extend(scores.cpu().numpy())
            all_matches.extend([False] * len(scores))
            continue
        # FIXED: Sort by scores descending and match greedily in that order
        sorted_idx = torch.argsort(scores, descending=True)
        pred_box_sorted = pred_box[sorted_idx]
        scores_sorted = scores[sorted_idx]
        gt_matched = set()
        for j in range(len(pred_box_sorted)):
            iou_matrix = box_iou_batch(pred_box_sorted[j : j + 1], gt_box)
            max_iou, gt_id = iou_matrix.max(dim=1)
            score = scores_sorted[j].item()
            all_scores.append(score)
            if max_iou.item() >= iou_threshold and gt_id.item() not in gt_matched:
                all_matches.append(True)
                gt_matched.add(gt_id.item())
            else:
                all_matches.append(False)
        total_gt += len(gt_box)
    if len(all_scores) == 0 or total_gt == 0:
        return 0.0, 0.0, 0.0
    # Global sort (though per-image is already sorted, global ensures consistency)
    sorted_indices = np.argsort(all_scores)[::-1]
    all_matches = np.array(all_matches)[sorted_indices]
    tp_cumsum = np.cumsum(all_matches)
    fp_cumsum = np.cumsum(~all_matches)
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    final_recall = recalls[-1] if len(recalls) > 0 else 0.0
    return ap, final_precision, final_recall


def compute_map_metrics(
    pred_boxes: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    iou_thresholds: List[float] = [0.25, 0.5, 0.75],
) -> Dict[str, float]:
    """
    Compute mAP metrics at multiple IoU thresholds.

    Args:
        pred_boxes: List of predicted boxes per image
        gt_boxes: List of ground truth boxes per image
        pred_scores: List of prediction scores per image
        iou_thresholds: List of IoU thresholds to evaluate

    Returns:
        Dictionary containing mAP@0.25, mAP@0.5, mAP@0.75, avg_iou, etc.
    """
    metrics = {}

    # Compute AP at different IoU thresholds
    for iou_thresh in iou_thresholds:
        ap, precision, recall = compute_ap_at_iou(
            pred_boxes, gt_boxes, pred_scores, iou_threshold=iou_thresh
        )
        metrics[f"mAP@{iou_thresh}"] = ap
        metrics[f"precision@{iou_thresh}"] = precision
        metrics[f"recall@{iou_thresh}"] = recall

    # Compute average IoU across all predictions
    all_ious = []
    for pred_box, gt_box in zip(pred_boxes, gt_boxes):
        if len(pred_box) > 0 and len(gt_box) > 0:
            iou_matrix = box_iou_batch(pred_box, gt_box)
            max_iou = iou_matrix.max(dim=1)[0]
            all_ious.extend(max_iou.cpu().numpy())

    metrics["avg_iou"] = np.mean(all_ious) if len(all_ious) > 0 else 0.0

    # Compute mAP averaged over multiple IoU thresholds (COCO-style mAP)
    if len(iou_thresholds) > 1:
        map_values = [metrics[f"mAP@{t}"] for t in iou_thresholds]
        metrics["mAP"] = np.mean(map_values)

    return metrics


def compute_detr_metrics(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    bbox_mask: torch.Tensor,
    num_queries: int = 100,
) -> Dict[str, float]:
    """
    Compute DETR-specific metrics for a batch.

    Args:
        pred_boxes: (B, num_queries, 4) predicted boxes
        gt_boxes: (B, max_gt, 4) ground truth boxes
        pred_scores: (B, num_queries) objectness scores
        bbox_mask: (B, max_gt) mask for valid ground truth boxes
        num_queries: Number of object queries

    Returns:
        Dictionary with mAP@0.5, mAP@0.25, avg_iou, etc.
    """
    batch_size = pred_boxes.size(0)

    pred_boxes_list = []
    gt_boxes_list = []
    pred_scores_list = []

    for i in range(batch_size):
        # Get valid ground truth boxes
        valid_gt = gt_boxes[i][bbox_mask[i] > 0.5]

        # Get predictions (sorted by objectness score)
        obj_scores = pred_scores[i]
        sorted_idx = torch.argsort(obj_scores, descending=True)

        # Take top predictions (e.g., top 100 or number of queries)
        top_k = min(num_queries, len(sorted_idx))
        top_pred_boxes = pred_boxes[i][sorted_idx[:top_k]]
        top_scores = obj_scores[sorted_idx[:top_k]]

        pred_boxes_list.append(top_pred_boxes)
        gt_boxes_list.append(valid_gt)
        pred_scores_list.append(top_scores)

    # Compute metrics
    metrics = compute_map_metrics(
        pred_boxes_list,
        gt_boxes_list,
        pred_scores_list,
        iou_thresholds=[0.25, 0.5, 0.75],
    )

    return metrics
