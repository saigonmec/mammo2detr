"""Hungarian matcher (Deformable-DETR style with focal classification cost)."""
from __future__ import annotations

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from ..utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("all matcher costs cannot be 0")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> list[tuple[Tensor, Tensor]]:
        """Return a list (per image) of ``(query_idx, target_idx)`` index tensors."""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        sizes = [len(t["boxes"]) for t in targets]
        device = outputs["pred_logits"].device
        empty = (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
        if sum(sizes) == 0:
            return [empty for _ in range(bs)]

        with torch.autocast(device_type=device.type, enabled=False):
            out_prob = outputs["pred_logits"].flatten(0, 1).float().sigmoid()  # [B*Q, C]
            out_bbox = outputs["pred_boxes"].flatten(0, 1).float()  # [B*Q, 4]
            tgt_ids = torch.cat([t["labels"] for t in targets])
            tgt_bbox = torch.cat([t["boxes"] for t in targets]).float()

            neg_cost = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()
            C = torch.nan_to_num(C, nan=1e5, posinf=1e5, neginf=-1e5)

        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            if sizes[i] == 0:
                indices.append(empty)
                continue
            row, col = linear_sum_assignment(c[i].numpy())
            indices.append((torch.as_tensor(row, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))
        return indices
