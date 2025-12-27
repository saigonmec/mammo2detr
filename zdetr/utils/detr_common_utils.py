"""
Common utilities for DETR model
- Box operations (IoU, conversion)
- Hungarian matching
- Loss computation
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def parse_img_size(val):
    """Parse image size from string or int"""
    if val is None:
        return None
    if isinstance(val, str) and "x" in val:
        h, w = val.lower().split("x")
        return (int(h), int(w))
    else:
        s = int(val)
        return (s, s)


def box_cxcywh_to_xyxy(boxes):
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes):
    """Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h]"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes
    Args:
        boxes1: [N, 4] in COCO format [x, y, w, h]
        boxes2: [M, 4] in COCO format [x, y, w, h]
    Returns:
        iou: [N, M]
    """
    # Convert COCO format [x, y, w, h] to [x1, y1, x2, y2]
    boxes1_xyxy = boxes1.clone()
    boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2]  # x2 = x + w
    boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3]  # y2 = y + h

    boxes2_xyxy = boxes2.clone()
    boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2]
    boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3]

    # Compute areas
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (
        boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]
    )
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (
        boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]
    )

    # Compute intersection
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Compute union and IoU
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)

    return iou


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU (GIoU) for DETR
    Args:
        boxes1, boxes2: [N, 4] or [M, 4] in COCO format [x, y, w, h]
    Returns:
        giou: [N, M]
    """
    # Convert to xyxy
    boxes1_xyxy = boxes1.clone()
    boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2]
    boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3]

    boxes2_xyxy = boxes2.clone()
    boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2]
    boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3]

    # Standard IoU
    iou = box_iou(boxes1, boxes2)

    # Compute enclosing box
    lt = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])
    rb = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (
        boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]
    )
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (
        boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]
    )
    union = area1[:, None] + area2 - iou * area1[:, None]

    giou = iou - (area_c - union) / (area_c + 1e-6)

    return giou


def hungarian_matching(pred_boxes, pred_scores, gt_boxes, cost_bbox=5.0, cost_giou=2.0):
    """
    Hungarian matching between predictions and ground truths
    Args:
        pred_boxes: [N_q, 4] predicted boxes in normalized COCO format
        pred_scores: [N_q, 1] objectness scores
        gt_boxes: [N_gt, 4] ground truth boxes in normalized COCO format
    Returns:
        pred_idx, gt_idx: matched indices
    """
    N_gt = gt_boxes.shape[0]

    if N_gt == 0:
        return [], []

    # L1 cost for bbox
    cost_bbox_val = torch.cdist(pred_boxes, gt_boxes, p=1)

    # GIoU cost (negative for maximization)
    giou = generalized_box_iou(pred_boxes, gt_boxes)
    cost_giou_val = -giou

    # Total cost
    C = cost_bbox * cost_bbox_val + cost_giou * cost_giou_val
    C = C.cpu().detach().numpy()

    # Hungarian algorithm
    pred_idx, gt_idx = linear_sum_assignment(C)

    return pred_idx.tolist(), gt_idx.tolist()


def compute_iou_weighted_obj_loss(pred_boxes, pred_scores, gt_boxes, dn_outputs=None):
    """
    Compute IoU-weighted objectness loss
    Args:
        pred_boxes: [B, N_q, 4]
        pred_scores: [B, N_q, 1]
        gt_boxes: [B, N_gt, 4]
        dn_outputs: optional denoising outputs
    Returns:
        loss: scalar
    """
    B = pred_boxes.shape[0]
    total_loss = 0.0

    for b in range(B):
        pred_b = pred_boxes[b]
        score_b = pred_scores[b]
        gt_b = gt_boxes[b]

        # Remove padding
        valid_gt = gt_b.sum(dim=-1) > 0
        gt_b = gt_b[valid_gt]

        if gt_b.shape[0] == 0:
            # No ground truth - all predictions should be negative
            target = torch.zeros_like(score_b)
            weight = torch.ones_like(score_b)
            loss_b = F.binary_cross_entropy_with_logits(score_b, target, weight=weight)
            total_loss += loss_b
            continue

        # Hungarian matching
        pred_idx, gt_idx = hungarian_matching(pred_b, score_b, gt_b)

        # Compute IoU-weighted loss
        target = torch.zeros_like(score_b)
        weight = torch.ones_like(score_b)

        for p_idx, g_idx in zip(pred_idx, gt_idx):
            iou = box_iou(pred_b[p_idx : p_idx + 1], gt_b[g_idx : g_idx + 1])[0, 0]
            target[p_idx] = 1.0
            weight[p_idx] = iou.clamp(min=0.01)

        loss_b = F.binary_cross_entropy_with_logits(score_b, target, weight=weight)
        total_loss += loss_b

    # Add denoising loss if provided
    if dn_outputs is not None and len(dn_outputs) > 0:
        for dn_output in dn_outputs:
            dn_pred_boxes = dn_output["pred_bboxes"]
            dn_pred_scores = dn_output["obj_scores"]
            dn_gt_boxes = dn_output["gt_boxes"]

            for b in range(B):
                dn_pred_b = dn_pred_boxes[b]
                dn_score_b = dn_pred_scores[b]
                dn_gt_b = dn_gt_boxes[b]

                valid_gt = dn_gt_b.sum(dim=-1) > 0
                dn_gt_b = dn_gt_b[valid_gt]
                dn_pred_b = dn_pred_b[valid_gt]
                dn_score_b = dn_score_b[valid_gt]

                if dn_gt_b.shape[0] == 0:
                    continue

                # All denoising queries are positive
                iou = box_iou(dn_pred_b, dn_gt_b).diagonal()
                target = torch.ones_like(dn_score_b)
                weight = iou.unsqueeze(1).clamp(min=0.01)

                loss_dn = F.binary_cross_entropy_with_logits(
                    dn_score_b, target, weight=weight
                )
                total_loss += loss_dn

    return total_loss / B


def bbox_to_pixel(bbox, original_size):
    """
    Convert normalized bbox to pixel coordinates
    Args:
        bbox: [4] in normalized COCO format [x, y, w, h]
        original_size: (height, width) tuple
    Returns:
        (x_pix, y_pix, w_pix, h_pix) in pixel coordinates
    """
    x, y, w, h = bbox
    orig_h, orig_w = original_size
    x_pix = x * orig_w
    y_pix = y * orig_h
    w_pix = w * orig_w
    h_pix = h * orig_h
    return x_pix, y_pix, w_pix, h_pix


def compute_bbox_confidence(pred_bbox, gt_bbox):
    """Compute IoU as bbox confidence between single boxes"""
    iou = box_iou(pred_bbox.unsqueeze(0), gt_bbox.unsqueeze(0))
    return iou.item()
