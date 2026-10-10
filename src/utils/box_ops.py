"""Bounding-box utilities.

Convention used everywhere in :mod:`src`:

* **Model / targets / losses**: normalized ``cxcywh`` (center x, center y,
  width, height) in ``[0, 1]`` relative to the *network input* size.
* **IoU / GIoU computations**: ``xyxy``.
* **Metadata CSV**: pixel COCO ``xywh`` (top-left x, top-left y, w, h) in the
  *original* image, which is converted exactly once in the dataset.

Keeping a single convention is the first fix over the legacy ``zdetr`` code,
which mixed three formats and therefore computed wrong GIoU losses and metrics.
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "box_cxcywh_to_xyxy",
    "box_xyxy_to_cxcywh",
    "box_xywh_to_xyxy",
    "box_xyxy_to_xywh",
    "box_area",
    "box_iou",
    "generalized_box_iou",
    "box_iou_pairwise",
    "generalized_box_iou_pairwise",
    "boxes_center_inside",
    "clip_boxes_xyxy",
    "scale_boxes",
    "inverse_sigmoid",
]


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def box_xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """COCO ``[x, y, w, h]`` -> ``[x0, y0, x1, y1]``."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def box_xyxy_to_xywh(boxes: Tensor) -> Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """Area of ``xyxy`` boxes (degenerate boxes give 0)."""
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def _check_xyxy(boxes: Tensor, name: str) -> None:
    if boxes.numel() == 0:
        return
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name}: non-finite box coordinates")
    if (boxes[:, 2:] < boxes[:, :2] - 1e-6).any():
        raise ValueError(f"{name}: expected xyxy boxes with x1>=x0 and y1>=y0")


def box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> tuple[Tensor, Tensor]:
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Returns ``(iou, union)`` each of shape ``[N, M]``.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=eps)
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Pairwise GIoU (https://giou.stanford.edu/) for ``xyxy`` boxes -> ``[N, M]``."""
    _check_xyxy(boxes1, "boxes1")
    _check_xyxy(boxes2, "boxes2")
    iou, union = box_iou(boxes1, boxes2, eps)
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area.clamp(min=eps)


def box_iou_pairwise(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Element-wise IoU of two ``[N, 4]`` xyxy tensors -> ``[N]``."""
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = box_area(boxes1) + box_area(boxes2) - inter
    return inter / union.clamp(min=eps)


def generalized_box_iou_pairwise(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Element-wise GIoU of two ``[N, 4]`` xyxy tensors -> ``[N]``."""
    iou = box_iou_pairwise(boxes1, boxes2, eps)
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, 0] * wh[:, 1]
    inter = (torch.min(boxes1[:, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, :2], boxes2[:, :2])).clamp(min=0)
    union = box_area(boxes1) + box_area(boxes2) - inter[:, 0] * inter[:, 1]
    return iou - (area - union) / area.clamp(min=eps)


def boxes_center_inside(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """``[N, M]`` bool: center of ``boxes1[i]`` lies inside ``boxes2[j]`` (xyxy)."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(boxes1.shape[0], boxes2.shape[0], dtype=torch.bool, device=boxes1.device)
    cx = (boxes1[:, 0] + boxes1[:, 2]) / 2
    cy = (boxes1[:, 1] + boxes1[:, 3]) / 2
    inside_x = (cx[:, None] >= boxes2[None, :, 0]) & (cx[:, None] <= boxes2[None, :, 2])
    inside_y = (cy[:, None] >= boxes2[None, :, 1]) & (cy[:, None] <= boxes2[None, :, 3])
    return inside_x & inside_y


def clip_boxes_xyxy(boxes: Tensor, height: float, width: float) -> Tensor:
    x0 = boxes[..., 0].clamp(0, width)
    y0 = boxes[..., 1].clamp(0, height)
    x1 = boxes[..., 2].clamp(0, width)
    y1 = boxes[..., 3].clamp(0, height)
    return torch.stack([x0, y0, x1, y1], dim=-1)


def scale_boxes(boxes: Tensor, scale_w: float, scale_h: float) -> Tensor:
    """Multiply x-coordinates by ``scale_w`` and y-coordinates by ``scale_h`` (any 4-format)."""
    s = boxes.new_tensor([scale_w, scale_h, scale_w, scale_h])
    return boxes * s


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))
