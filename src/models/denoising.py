"""Contrastive denoising (CDN) queries, following DN-DETR / DINO.

For every ground-truth box we create ``num_groups`` noised copies (and, with
``contrastive=True``, an additional *negative* copy per group with larger
noise that must be classified as background). The noised boxes are fed to the
decoder as extra queries with a known assignment, which gives the model a
direct box-regression signal from epoch 0 and stabilises Hungarian matching.
A block-diagonal self-attention mask prevents any information leak from the
denoising queries into the matching queries (the legacy code lacked this).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ..config import DenoisingConfig
from ..utils.box_ops import inverse_sigmoid


@dataclass
class DNMeta:
    pad_size: int  # number of denoising queries (per image, padded)
    num_groups: int
    single_pad: int  # max number of GT boxes in the batch
    contrastive: bool

    @property
    def group_size(self) -> int:
        return self.single_pad * (2 if self.contrastive else 1)

    def positive_slots(self, num_gt: int, group: int) -> Tensor:
        return torch.arange(num_gt) + group * self.group_size


def build_dn_queries(
    targets: list[dict[str, Tensor]],
    cfg: DenoisingConfig,
    num_classes: int,
    label_enc: nn.Embedding,
    num_matching_queries: int,
    num_extra_queries: int = 0,
) -> tuple[Tensor, Tensor, Tensor, DNMeta] | None:
    """Return ``(dn_content [B, pad, C], dn_boxes_unsig [B, pad, 4], attn_mask [T, T], meta)``
    or ``None`` when the batch contains no boxes.

    Query layout along the sequence: ``[DN groups ... | matching queries | extra queries]``.
    ``attn_mask`` follows ``nn.MultiheadAttention`` semantics (``True`` = not allowed).
    """
    device = label_enc.weight.device
    num_gts = [len(t["labels"]) for t in targets]
    max_gt = max(num_gts) if num_gts else 0
    if max_gt == 0:
        return None
    bs = len(targets)
    G = cfg.num_groups
    meta = DNMeta(pad_size=0, num_groups=G, single_pad=max_gt, contrastive=cfg.contrastive)
    meta.pad_size = G * meta.group_size

    labels = torch.cat([t["labels"] for t in targets]).to(device)
    boxes = torch.cat([t["boxes"] for t in targets]).to(device).float()  # cxcywh
    batch_idx = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_gts)]).to(device)
    repeat = 2 * G if cfg.contrastive else G
    known_labels = labels.repeat(repeat)
    known_boxes = boxes.repeat(repeat, 1)
    known_bid = batch_idx.repeat(repeat)
    n_known = len(labels)
    # slot index inside the image: [0..n_b) for each group; negatives in the second half of a group
    local_idx = torch.cat([torch.arange(n, device=device) for n in num_gts])
    local_idx = local_idx.repeat(repeat)
    group_idx = torch.arange(repeat, device=device).repeat_interleave(n_known)
    if cfg.contrastive:
        # copies alternate: [pos g0, neg g0, pos g1, neg g1, ...]
        is_negative = (group_idx % 2 == 1)
        group_of = group_idx // 2
        slot = group_of * meta.group_size + is_negative.long() * meta.single_pad + local_idx
    else:
        is_negative = torch.zeros_like(group_idx, dtype=torch.bool)
        slot = group_idx * meta.group_size + local_idx

    # ---- label noise (content queries)
    noised_labels = known_labels.clone()
    if cfg.label_noise_ratio > 0:
        p = torch.rand_like(noised_labels.float())
        chosen = p < cfg.label_noise_ratio * 0.5
        new_labels = torch.randint_like(noised_labels, 0, num_classes)
        noised_labels = torch.where(chosen, new_labels, noised_labels)

    # ---- box noise
    noised_boxes = known_boxes.clone()
    if cfg.box_noise_scale > 0:
        xyxy = torch.cat([known_boxes[:, :2] - known_boxes[:, 2:] / 2, known_boxes[:, :2] + known_boxes[:, 2:] / 2], 1)
        diff = torch.cat([known_boxes[:, 2:] / 2, known_boxes[:, 2:] / 2], 1)
        rand_sign = torch.randint_like(known_boxes, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(known_boxes)
        rand_part[is_negative] += 1.0
        rand_part = rand_part * rand_sign
        xyxy = (xyxy + rand_part * diff * cfg.box_noise_scale).clamp(0.0, 1.0)
        noised_boxes = torch.cat([(xyxy[:, :2] + xyxy[:, 2:]) / 2, (xyxy[:, 2:] - xyxy[:, :2]).clamp(min=1e-4)], 1)

    content = label_enc(noised_labels)  # [K, C]
    dn_content = torch.zeros(bs, meta.pad_size, content.shape[-1], device=device, dtype=content.dtype)
    dn_boxes = torch.zeros(bs, meta.pad_size, 4, device=device)
    dn_content[known_bid, slot] = content
    dn_boxes[known_bid, slot] = noised_boxes
    dn_boxes_unsig = inverse_sigmoid(dn_boxes)
    # Unused (padded) slots keep a harmless reference box at the image centre.
    pad_mask = torch.ones(bs, meta.pad_size, dtype=torch.bool, device=device)
    pad_mask[known_bid, slot] = False
    dn_boxes_unsig = torch.where(pad_mask[..., None], torch.zeros_like(dn_boxes_unsig), dn_boxes_unsig)

    # ---- self-attention mask
    total = meta.pad_size + num_matching_queries + num_extra_queries
    attn_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    attn_mask[meta.pad_size :, : meta.pad_size] = True  # matching/extra queries cannot see DN queries
    gs = meta.group_size
    for g in range(G):
        s, e = g * gs, (g + 1) * gs
        attn_mask[s:e, :s] = True
        attn_mask[s:e, e : meta.pad_size] = True
    return dn_content, dn_boxes_unsig, attn_mask, meta


def dn_matched_indices(meta: DNMeta, targets: list[dict[str, Tensor]]) -> list[tuple[Tensor, Tensor]]:
    """Known assignment between *positive* DN slots and ground-truth boxes."""
    out = []
    for t in targets:
        n = len(t["labels"])
        if n == 0:
            out.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
            continue
        src = torch.cat([meta.positive_slots(n, g) for g in range(meta.num_groups)])
        tgt = torch.arange(n).repeat(meta.num_groups)
        out.append((src, tgt))
    return out
