"""Turn raw model outputs into per-image detections in original-image pixels."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import batched_nms

from ..utils.box_ops import box_cxcywh_to_xyxy


class PostProcess(nn.Module):
    def __init__(self, topk: int = 10, score_threshold: float = 0.0, nms_iou: float | None = None):
        super().__init__()
        self.topk = topk
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]) -> list[dict[str, Tensor]]:
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        prob = logits.sigmoid()
        B, Q, C = prob.shape
        k = min(self.topk, Q * C)
        scores, idx = prob.view(B, -1).topk(k, dim=1)
        q_idx = torch.div(idx, C, rounding_mode="floor")
        labels = idx % C
        sel = torch.gather(boxes, 1, q_idx.unsqueeze(-1).repeat(1, 1, 4))
        xyxy = box_cxcywh_to_xyxy(sel)
        results = []
        for i in range(B):
            vh, vw = targets[i]["size"].tolist()
            oh, ow = targets[i]["orig_size"].tolist()
            # canvas (valid region) pixels -> original pixels
            scale = torch.tensor([vw * ow / vw, vh * oh / vh, vw * ow / vw, vh * oh / vh], device=xyxy.device)
            b = xyxy[i] * scale
            b[:, 0::2] = b[:, 0::2].clamp(0, ow)
            b[:, 1::2] = b[:, 1::2].clamp(0, oh)
            s, l, q = scores[i], labels[i], q_idx[i]
            keep = s >= self.score_threshold
            b, s, l, q = b[keep], s[keep], l[keep], q[keep]
            if self.nms_iou is not None and len(b) > 1:
                keep = batched_nms(b.float(), s.float(), l, self.nms_iou)
                b, s, l, q = b[keep], s[keep], l[keep], q[keep]
            results.append({"scores": s, "labels": l, "boxes": b, "query_idx": q})
        return results
