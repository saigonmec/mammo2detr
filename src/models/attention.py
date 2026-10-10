"""Multi-Scale Deformable Attention (Zhu et al., 2021) in pure PyTorch.

A faithful port of the reference ``ms_deform_attn_core_pytorch`` so the model
runs on CPU / MPS / CUDA without compiling custom kernels.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def ms_deform_attn_core_pytorch(
    value: Tensor,
    value_spatial_shapes: list[tuple[int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """``value``: [N, S, M, D]; ``sampling_locations``: [N, Lq, M, L, P, 2] in [0, 1];
    ``attention_weights``: [N, Lq, M, L, P] -> output [N, Lq, M*D]."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in value_spatial_shapes], dim=1)
    sampling_grids = (2 * sampling_locations - 1).to(value.dtype)
    sampled = []
    for lid, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[lid].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, h, w)
        grid_l = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)  # [N*M, Lq, P, 2]
        sampled.append(F.grid_sample(value_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False))
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_).to(value.dtype)
    output = (torch.stack(sampled, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model, self.n_levels, self.n_heads, self.n_points = d_model, n_levels, n_heads, n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.last_sampling: tuple[Tensor, Tensor] | None = None
        self.store_sampling = False
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: list[tuple[int, int]],
        input_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """``query`` [N, Lq, C]; ``reference_points`` [N, Lq, L, 2|4] (normalized, per level);
        ``input_flatten`` [N, S, C]; ``input_padding_mask`` [N, S] (True = pad)."""
        N, Lq, _ = query.shape
        N, S, _ = input_flatten.shape
        assert sum(h * w for h, w in input_spatial_shapes) == S
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(N, S, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(query).view(N, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        attn = self.attention_weights(query).view(N, Lq, self.n_heads, self.n_levels * self.n_points)
        attn = F.softmax(attn, -1).view(N, Lq, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 2:
            normalizer = torch.tensor([[w, h] for h, w in input_spatial_shapes], dtype=query.dtype, device=query.device)
            locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")
        if self.store_sampling:
            self.last_sampling = (locations.detach(), attn.detach())
        out = ms_deform_attn_core_pytorch(value, input_spatial_shapes, locations, attn)
        return self.output_proj(out)


class ConditionalCrossAttention(nn.Module):
    """Cross-attention with *decoupled* content and positional terms
    (Conditional-DETR / DAB-DETR).

    Queries and keys are the concatenation ``[content ; position]`` so the
    attention logit is ``content·content + position·position``: a query whose
    anchor box sits at ``(x, y)`` attends to the feature at ``(x, y)`` from the
    very first iteration, without having to *learn* that alignment through
    shared projections (the reason vanilla additive DETR attention converges
    slowly, and the reason denoising queries cannot use their box otherwise).
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_content = nn.Linear(d_model, d_model)
        self.q_pos = nn.Linear(d_model, d_model)
        self.k_content = nn.Linear(d_model, d_model)
        self.k_pos = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.init_positional_projections()

    @torch.no_grad()
    def init_positional_projections(self, gain: float = 1.4142) -> None:
        """Initialise ``q_pos`` and ``k_pos`` with the *same* permutation matrix.

        The positional logit then starts as ``sine(anchor) . sine(position)``,
        i.e. a kernel peaked where the key position equals the anchor centre,
        with the (sin, cos) frequency pairs of both coordinates distributed
        round-robin over the heads. Attention is therefore localised around
        the anchor box from the first iteration (the same inductive bias that
        deformable sampling has for free) while remaining fully learnable.
        """
        d, H = self.q_pos.in_features, self.nhead
        D = d // H
        if D % 4 == 0 and d % 4 == 0:
            W = torch.zeros(d, d)
            for h in range(H):
                for s in range(D):
                    coord = 0 if s < D // 2 else 1  # y-part then x-part of the sine embedding
                    local = s - (D // 2) * coord
                    pair = h + H * (local // 2)  # round-robin over frequencies
                    comp = local % 2
                    W[h * D + s, coord * (d // 2) + 2 * pair + comp] = 1.0
        else:
            W = torch.eye(d)
        self.q_pos.weight.copy_(W * gain)
        self.k_pos.weight.copy_(W * gain)
        self.q_pos.bias.zero_()
        self.k_pos.bias.zero_()

    def forward(
        self,
        tgt: Tensor,
        query_sine: Tensor,
        memory: Tensor,
        memory_pos: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """``tgt`` [B, Nq, C] content; ``query_sine`` [B, Nq, C] positional (sine of the anchor,
        optionally pre-scaled); ``memory``/``memory_pos`` [B, S, C]; ``key_padding_mask`` [B, S]."""
        B, Nq, C = tgt.shape
        S = memory.shape[1]
        H, D = self.nhead, self.head_dim
        qc = self.q_content(tgt).view(B, Nq, H, D)
        qp = self.q_pos(query_sine).view(B, Nq, H, D)
        kc = self.k_content(memory).view(B, S, H, D)
        kp = self.k_pos(memory_pos).view(B, S, H, D)
        v = self.v_proj(memory).view(B, S, H, D).transpose(1, 2)
        q = torch.cat([qc, qp], dim=-1).transpose(1, 2)  # [B, H, Nq, 2D]
        k = torch.cat([kc, kp], dim=-1).transpose(1, 2)  # [B, H, S, 2D]
        weights = None
        if need_weights:
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(2 * D)
            if key_padding_mask is not None:
                logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            w = logits.softmax(-1)
            weights = w.mean(1).detach()
            out = F.dropout(w, self.dropout, self.training) @ v
        else:
            attn_mask = None if key_padding_mask is None else (~key_padding_mask)[:, None, None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, Nq, C)
        return self.out_proj(out), weights
