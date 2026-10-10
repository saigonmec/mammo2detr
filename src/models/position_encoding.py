"""Sine positional encodings for feature maps (DETR) and for boxes (DAB-DETR)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionEmbeddingSine(nn.Module):
    """2-D sine/cosine positional encoding computed from a padding mask.

    ``forward(mask)`` with ``mask[B, H, W]`` (``True`` = padding) returns
    ``[B, 2*num_pos_feats, H, W]``. Padding-aware normalisation keeps the
    encoding consistent between letterboxed and un-padded images.
    """

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0, normalize: bool = True, scale: float | None = None, offset: float = -0.5):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale
        self.offset = offset

    def forward(self, mask: Tensor) -> Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def gen_sineembed_for_position(pos_tensor: Tensor, num_pos_feats: int = 128, temperature: float = 10000.0) -> Tensor:
    """Sine embedding of normalized points/boxes ``[B, N, 2|4]`` (DAB-DETR).

    Returns ``[B, N, 2*num_pos_feats]`` for points and ``[B, N, 4*num_pos_feats]``
    for boxes (order: y, x, w, h).
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def embed(v: Tensor) -> Tensor:
        v = v.float() * scale
        e = v[:, :, None] / dim_t
        return torch.stack((e[:, :, 0::2].sin(), e[:, :, 1::2].cos()), dim=3).flatten(2)

    pos_x = embed(pos_tensor[:, :, 0])
    pos_y = embed(pos_tensor[:, :, 1])
    if pos_tensor.size(-1) == 2:
        return torch.cat((pos_y, pos_x), dim=2)
    if pos_tensor.size(-1) == 4:
        pos_w = embed(pos_tensor[:, :, 2])
        pos_h = embed(pos_tensor[:, :, 3])
        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    raise ValueError(f"Unknown pos_tensor shape(-1): {pos_tensor.size(-1)}")
