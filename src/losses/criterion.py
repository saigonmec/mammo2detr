"""Set criterion for MammoDETR: Hungarian-matched detection losses (+ aux, + DN)
and image-level classification losses."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import Config
from ..utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou_pairwise
from ..utils.misc import get_world_size, is_dist
from .focal import build_image_loss, sigmoid_focal_loss
from .matcher import HungarianMatcher


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_det_classes: int,
        matcher: HungarianMatcher,
        det_weights: dict[str, float],
        focal_alpha: float,
        focal_gamma: float,
        image_losses: nn.ModuleDict,
        image_weights: dict[str, float],
        aux_weight: float = 1.0,
        dn_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_det_classes
        self.matcher = matcher
        self.det_weights = det_weights  # {"loss_ce", "loss_bbox", "loss_giou"}
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.image_losses = image_losses
        self.image_weights = image_weights
        self.aux_weight = aux_weight
        self.dn_weight = dn_weight

    # ------------------------------------------------------------------ #
    @staticmethod
    def _src_idx(indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs: dict, targets: list[dict], indices, num_boxes: float) -> dict[str, Tensor]:
        logits = outputs["pred_logits"]
        idx = self._src_idx(indices)
        target_classes = torch.full(logits.shape[:2], self.num_classes, dtype=torch.int64, device=logits.device)
        if idx[0].numel() > 0:
            target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)]).to(logits.device)
            target_classes[idx] = target_classes_o
        onehot = torch.zeros(logits.shape[0], logits.shape[1], logits.shape[2] + 1, dtype=logits.dtype, device=logits.device)
        onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        onehot = onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(logits, onehot, num_boxes, self.alpha, self.gamma) * logits.shape[1]
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs: dict, targets: list[dict], indices, num_boxes: float) -> dict[str, Tensor]:
        idx = self._src_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        if src_boxes.numel() == 0:
            z = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": z, "loss_giou": z}
        tgt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)]).to(src_boxes)
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes
        giou = generalized_box_iou_pairwise(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
        loss_giou = (1 - giou).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def det_losses(self, outputs: dict, targets: list[dict], indices, num_boxes: float) -> dict[str, Tensor]:
        d = {}
        d.update(self.loss_labels(outputs, targets, indices, num_boxes))
        d.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return d

    # ------------------------------------------------------------------ #
    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, Tensor]:
        device = outputs["pred_logits"].device
        num_boxes = torch.as_tensor(float(sum(len(t["labels"]) for t in targets)), device=device)
        if is_dist():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = float(torch.clamp(num_boxes / get_world_size(), min=1).item())

        losses: dict[str, Tensor] = {}
        total = torch.zeros((), device=device)

        # main detection losses
        indices = self.matcher(outputs, targets)
        main = self.det_losses(outputs, targets, indices, num_boxes)
        for k, v in main.items():
            losses[k] = v
            total = total + self.det_weights[k] * v
        losses["matched_queries"] = torch.as_tensor(float(sum(len(i) for i, _ in indices)), device=device)

        # auxiliary decoder layers
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            aux_idx = self.matcher(aux, targets)
            for k, v in self.det_losses(aux, targets, aux_idx, num_boxes).items():
                losses[f"{k}_aux{i}"] = v
                total = total + self.aux_weight * self.det_weights[k] * v

        # denoising queries (known assignment)
        dn_out = outputs.get("dn_outputs")
        if dn_out is not None:
            from ..models.denoising import dn_matched_indices

            meta = outputs["dn_meta"]
            dn_idx = dn_matched_indices(meta, targets)
            dn_num_boxes = num_boxes * meta.num_groups
            for k, v in self.det_losses(dn_out, targets, dn_idx, dn_num_boxes).items():
                losses[f"{k}_dn"] = v
                total = total + self.dn_weight * self.det_weights[k] * v
            for i, aux in enumerate(dn_out.get("aux_outputs", [])):
                for k, v in self.det_losses(aux, targets, dn_idx, dn_num_boxes).items():
                    losses[f"{k}_dn_aux{i}"] = v
                    total = total + self.dn_weight * self.aux_weight * self.det_weights[k] * v

        # image-level targets
        if "image_logits" in outputs:
            for name, logits in outputs["image_logits"].items():
                y = torch.stack([t["image_labels"][name] for t in targets]).to(device).long()
                v = self.image_losses[name](logits.float(), y)
                losses[f"loss_img_{name}"] = v
                total = total + self.image_weights[name] * v

        losses["loss_total"] = total
        return losses


def build_criterion(cfg: Config, image_class_counts: dict[str, list[int]] | None = None) -> SetCriterion:
    matcher = HungarianMatcher(
        cost_class=cfg.loss.matcher_cost_class,
        cost_bbox=cfg.loss.matcher_cost_bbox,
        cost_giou=cfg.loss.matcher_cost_giou,
        focal_alpha=cfg.loss.focal_alpha,
        focal_gamma=cfg.loss.focal_gamma,
    )
    det_weights = {"loss_ce": cfg.loss.det_class_weight, "loss_bbox": cfg.loss.bbox_weight, "loss_giou": cfg.loss.giou_weight}
    image_losses = nn.ModuleDict()
    image_weights: dict[str, float] = {}
    if cfg.model.heads.image_cls != "none":
        for it in cfg.data.image_labels:
            counts = (image_class_counts or {}).get(it.name)
            image_losses[it.name] = build_image_loss(it.loss, it.num_classes, counts, it.class_weights, it.label_smoothing, it.focal_gamma)
            image_weights[it.name] = cfg.loss.image_loss_weight * it.weight
    return SetCriterion(
        num_det_classes=len(cfg.data.det_class_names),
        matcher=matcher,
        det_weights=det_weights,
        focal_alpha=cfg.loss.focal_alpha,
        focal_gamma=cfg.loss.focal_gamma,
        image_losses=image_losses,
        image_weights=image_weights,
        aux_weight=cfg.loss.aux_weight,
        dn_weight=cfg.loss.dn_weight,
    )
