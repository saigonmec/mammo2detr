"""Classification losses: detection focal loss and image-level losses."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def sigmoid_focal_loss(
    inputs: Tensor, targets: Tensor, num_boxes: float, alpha: float = 0.25, gamma: float = 2.0
) -> Tensor:
    """Sigmoid focal loss as used by Deformable-DETR / DINO.

    ``inputs``/``targets``: ``[B, Q, C]``; the loss is averaged over queries,
    summed over the batch and classes, and divided by ``num_boxes``.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


class MulticlassFocalLoss(nn.Module):
    """Focal loss for softmax classification with optional class weights and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        logp = F.log_softmax(logits, dim=-1)
        num_classes = logits.shape[-1]
        with torch.no_grad():
            true = torch.full_like(logp, self.label_smoothing / max(num_classes - 1, 1))
            true.scatter_(1, targets.view(-1, 1), 1.0 - self.label_smoothing)
        p = logp.exp()
        focal = (1.0 - p) ** self.gamma
        loss = -(true * focal * logp)
        if self.weight is not None:
            loss = loss * self.weight.to(logits.dtype)[None, :]
        loss = loss.sum(-1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss (Cao et al., 2019)."""

    def __init__(self, cls_num_list: Sequence[int], max_m: float = 0.5, weight: Tensor | None = None, s: float = 30.0):
        super().__init__()
        n = torch.as_tensor(list(cls_num_list), dtype=torch.float32).clamp(min=1)
        m_list = 1.0 / torch.sqrt(torch.sqrt(n))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", m_list)
        self.register_buffer("weight", weight if weight is not None else None)
        self.s = float(s)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        batch_m = self.m_list[targets].to(logits.dtype)
        logits_m = logits.clone()
        logits_m[torch.arange(logits.shape[0], device=logits.device), targets] -= batch_m
        w = None if self.weight is None else self.weight.to(logits.dtype)
        return F.cross_entropy(self.s * logits_m, targets, weight=w)


def compute_class_weights(counts: Sequence[int], mode: str | Sequence[float] | None) -> Tensor | None:
    """``balanced``: N/(C*n_c); ``sqrt``: sqrt of balanced; list: explicit; None: no weights."""
    if mode is None:
        return None
    if not isinstance(mode, str):
        return torch.as_tensor(list(mode), dtype=torch.float32)
    counts_t = torch.as_tensor(list(counts), dtype=torch.float32).clamp(min=1)
    balanced = counts_t.sum() / (len(counts_t) * counts_t)
    if mode == "balanced":
        return balanced
    if mode == "sqrt":
        return balanced.sqrt()
    raise ValueError(f"Unknown class_weights mode: {mode}")


def build_image_loss(loss_name: str, num_classes: int, class_counts: Sequence[int] | None, class_weights, label_smoothing: float = 0.0, focal_gamma: float = 2.0) -> nn.Module:
    counts = list(class_counts) if class_counts is not None else [1] * num_classes
    if len(counts) < num_classes:
        counts = counts + [1] * (num_classes - len(counts))
    weight = compute_class_weights(counts, class_weights)
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    if loss_name == "focal":
        return MulticlassFocalLoss(gamma=focal_gamma, weight=weight, label_smoothing=label_smoothing)
    if loss_name == "ldam":
        return LDAMLoss(counts, weight=weight)
    raise ValueError(f"Unknown image loss '{loss_name}' (use ce|focal|ldam)")
