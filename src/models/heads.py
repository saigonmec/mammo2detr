"""Prediction heads: per-layer detection heads and image-level classifiers."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor

from .layers import MLP, bias_init_with_prob


class DetectionHead(nn.Module):
    """Per-decoder-layer class logits (sigmoid/focal) and box deltas."""

    def __init__(self, d_model: int, num_classes: int, num_layers: int, prior_prob: float = 0.01, share: bool = False):
        super().__init__()
        class_embed = nn.Linear(d_model, num_classes)
        nn.init.constant_(class_embed.bias, bias_init_with_prob(prior_prob))
        bbox_embed = MLP(d_model, d_model, 4, 3)
        nn.init.constant_(bbox_embed.layers[-1].weight, 0.0)
        nn.init.constant_(bbox_embed.layers[-1].bias, 0.0)
        if share:
            self.class_embed = nn.ModuleList([class_embed] * num_layers)
            self.bbox_embed = nn.ModuleList([bbox_embed] * num_layers)
        else:
            self.class_embed = nn.ModuleList(copy.deepcopy(class_embed) for _ in range(num_layers))
            self.bbox_embed = nn.ModuleList(copy.deepcopy(bbox_embed) for _ in range(num_layers))


class ImageClassificationHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def masked_mean(tokens: Tensor, pad_mask: Tensor | None) -> Tensor:
    """Mean over valid tokens: ``tokens`` [B, S, C], ``pad_mask`` [B, S] (True = pad)."""
    if pad_mask is None:
        return tokens.mean(1)
    valid = (~pad_mask).to(tokens.dtype)[..., None]
    return (tokens * valid).sum(1) / valid.sum(1).clamp(min=1.0)
