"""Generic building blocks."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLP(nn.Module):
    """Simple multi-layer perceptron (FFN) as used for DETR box heads."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def get_activation_fn(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {name}")


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList(copy.deepcopy(module) for _ in range(n))


def bias_init_with_prob(prior_prob: float) -> float:
    """Bias value so that ``sigmoid(bias) == prior_prob`` (RetinaNet/focal init)."""
    return float(-math.log((1 - prior_prob) / prior_prob))


class FFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, activation: str = "relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
