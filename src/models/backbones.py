"""Backbones built on ``timm``.

Two adapters cover every architecture used in the legacy code base:

* :class:`TimmFeaturesBackbone` - CNNs and hierarchical transformers
  (ResNet, ResNeSt, RegNet, ConvNeXt(V2), EfficientNet, MaxViT, Swin, ...)
  through ``features_only=True``; levels are selected by output stride.
* :class:`TimmViTBackbone` - plain ViTs (DINOv2/v3, EVA02, ...) through
  ``forward_intermediates`` with correct handling of CLS/register tokens and
  dynamic image sizes, plus an optional ViTDet-style simple feature pyramid
  when several strides are requested.

Both return ``(features, masks)``: lists of ``[B, C_l, H_l, W_l]`` tensors and
``[B, H_l, W_l]`` boolean padding masks (``True`` = padding).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.misc import FrozenBatchNorm2d

from ..config import BackboneConfig
from ..utils.misc import downsample_mask

logger = logging.getLogger(__name__)

# Short aliases (kept compatible with the legacy ``zdetr`` names) -> timm ids.
BACKBONE_ALIASES: dict[str, str] = {
    # ResNet family
    "resnet18": "resnet18.a1_in1k",
    "resnet34": "resnet34.a1_in1k",
    "resnet50": "resnet50.a1_in1k",
    "resnet101": "resnet101.a1h_in1k",
    "resnet152": "resnet152.a1h_in1k",
    "resnext50": "resnext50_32x4d.a1h_in1k",
    "resnest50": "resnest50d.in1k",
    "resnest101": "resnest101e.in1k",
    "resnest50s2": "resnest50d_4s2x40d.in1k",
    "regnety": "regnety_080_tv.tv2_in1k",
    # ConvNeXt
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnextv2base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "dinov3_convnext_tiny": "convnext_tiny.dinov3_lvd1689m",
    "dinov3_convnext_small": "convnext_small.dinov3_lvd1689m",
    "dinov3_convnext_base": "convnext_base.dinov3_lvd1689m",
    "dinov3_convnext_large": "convnext_large.dinov3_lvd1689m",
    # EfficientNet
    "efficientnetv2": "efficientnetv2_rw_m.agc_in1k",
    "efficientnetv2s": "efficientnetv2_rw_s.ra2_in1k",
    # Attention CNN hybrids
    "maxvit_tiny": "maxvit_tiny_tf_224.in1k",
    "maxvit_tiny512": "maxvit_tiny_tf_512.in1k",
    "maxvit_small": "maxvit_small_tf_224.in1k",
    "maxvit_base": "maxvit_base_tf_224.in1k",
    "swinv2_tiny": "swinv2_tiny_window8_256.ms_in1k",
    "swinv2_small": "swinv2_small_window8_256.ms_in1k",
    "swinv2_base": "swinv2_base_window8_256.ms_in1k",
    "mambaout_tiny": "mambaout_tiny.in1k",
    # Plain ViTs
    "vit_small": "vit_small_patch14_reg4_dinov2.lvd142m",
    "eva02_small": "eva02_small_patch14_224.mim_in22k",
    "eva02_base": "eva02_base_patch14_448.mim_in22k_ft_in1k",
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_base": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_small_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_base_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov3_vit16small": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_vit16smallplus": "vit_small_plus_patch16_dinov3.lvd1689m",
    "dinov3_vit16base": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_vit16large": "vit_large_patch16_dinov3.lvd1689m",
}


def resolve_timm_name(name: str) -> str:
    if name.startswith("timm:"):
        return name[len("timm:") :]
    return BACKBONE_ALIASES.get(name, name)


# --------------------------------------------------------------------------- #
# Freezing helpers
# --------------------------------------------------------------------------- #
def convert_frozen_batchnorm(module: nn.Module) -> nn.Module:
    """Recursively replace ``BatchNorm2d`` with ``FrozenBatchNorm2d`` keeping statistics."""
    if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.affine:
        frozen = FrozenBatchNorm2d(module.num_features, eps=module.eps)
        frozen.weight.copy_(module.weight.detach())
        frozen.bias.copy_(module.bias.detach())
        if module.running_mean is not None:
            frozen.running_mean.copy_(module.running_mean)
            frozen.running_var.copy_(module.running_var)
        return frozen
    for name, child in module.named_children():
        module.add_module(name, convert_frozen_batchnorm(child))
    return module


def _set_requires_grad(module: nn.Module, flag: bool) -> int:
    n = 0
    for p in module.parameters():
        p.requires_grad_(flag)
        n += p.numel()
    return n


# --------------------------------------------------------------------------- #
# CNN / hierarchical backbone
# --------------------------------------------------------------------------- #
class TimmFeaturesBackbone(nn.Module):
    def __init__(self, timm_name: str, cfg: BackboneConfig):
        super().__init__()
        kwargs: dict[str, Any] = dict(cfg.timm_kwargs)
        if cfg.drop_path_rate > 0:
            kwargs.setdefault("drop_path_rate", cfg.drop_path_rate)
        self.model = timm.create_model(timm_name, pretrained=cfg.pretrained, features_only=True, **kwargs)
        reductions = list(self.model.feature_info.reduction())
        channels = list(self.model.feature_info.channels())
        self.level_indices: list[int] = []
        for s in cfg.out_strides:
            if s not in reductions:
                raise ValueError(
                    f"Backbone '{timm_name}' has feature strides {reductions}; requested stride {s}. "
                    "Adjust model.backbone.out_strides."
                )
            self.level_indices.append(reductions.index(s))
        self.out_channels = [channels[i] for i in self.level_indices]
        self.out_strides = [reductions[i] for i in self.level_indices]
        self._all_channels = channels
        self._freeze(cfg)

    def _freeze(self, cfg: BackboneConfig) -> None:
        if cfg.freeze_bn:
            convert_frozen_batchnorm(self.model)
        if cfg.freeze_at <= 0:
            return
        module_names = list(self.model.feature_info.module_name())
        stage_names = [n for n in module_names if not any(k in n for k in ("stem", "conv1", "act1", "bn1"))]
        children = list(self.model.named_children())
        frozen = 0
        # everything before the first stage counts as stem
        first_stage = stage_names[0].split(".")[0] if stage_names else None
        to_freeze: set[str] = set()
        if first_stage is not None:
            for name, _ in children:
                if name == first_stage:
                    break
                to_freeze.add(name)
        for n in stage_names[: cfg.freeze_at]:
            to_freeze.add(n.split(".")[0])
        for name, child in children:
            if name in to_freeze:
                frozen += _set_requires_grad(child, False)
        logger.info("Backbone: froze %d params in modules %s", frozen, sorted(to_freeze))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[list[Tensor], list[Tensor]]:
        feats = self.model(x)
        out, masks = [], []
        for li, ch in zip(self.level_indices, self.out_channels):
            f = feats[li]
            if f.ndim == 4 and f.shape[1] != ch and f.shape[-1] == ch:  # NHWC (e.g. Swin)
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
            if mask is None:
                masks.append(torch.zeros(f.shape[0], f.shape[2], f.shape[3], dtype=torch.bool, device=f.device))
            else:
                masks.append(downsample_mask(mask, (f.shape[2], f.shape[3])))
        return out, masks


# --------------------------------------------------------------------------- #
# Plain ViT backbone (+ simple feature pyramid)
# --------------------------------------------------------------------------- #
class SimpleFeaturePyramid(nn.Module):
    """ViTDet-style pyramid from a single-stride ViT map (scales 0.5x, 1x, 2x, 4x)."""

    def __init__(self, in_channels: int, scales: list[float]):
        super().__init__()
        self.scales = scales
        self.ops = nn.ModuleList()
        for s in scales:
            if math.isclose(s, 0.5):
                self.ops.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2))
            elif math.isclose(s, 1.0):
                self.ops.append(nn.Identity())
            elif math.isclose(s, 2.0):
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif math.isclose(s, 4.0):
                self.ops.append(nn.MaxPool2d(kernel_size=4, stride=4))
            else:
                raise ValueError(f"Unsupported ViT pyramid scale {s}; use strides patch/2, patch, 2*patch, 4*patch")

    def forward(self, x: Tensor) -> list[Tensor]:
        return [op(x) for op in self.ops]


def _is_plain_vit(model: nn.Module) -> bool:
    return hasattr(model, "blocks") and hasattr(model, "patch_embed") and not hasattr(model, "stages") and hasattr(model, "forward_intermediates")


class TimmViTBackbone(nn.Module):
    def __init__(self, model: nn.Module, timm_name: str, cfg: BackboneConfig):
        super().__init__()
        self.model = model
        ps = model.patch_embed.patch_size
        self.patch_stride = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        self.embed_dim = int(model.embed_dim)
        scales = [s / self.patch_stride for s in cfg.out_strides]
        self.pyramid = SimpleFeaturePyramid(self.embed_dim, scales)
        self.out_channels = [self.embed_dim] * len(scales)
        self.out_strides = [int(round(self.patch_stride * s)) for s in scales]
        self._freeze(cfg)

    def _freeze(self, cfg: BackboneConfig) -> None:
        if cfg.unfreeze_last_blocks is None:
            return
        n = cfg.unfreeze_last_blocks
        _set_requires_grad(self.model, False)
        blocks = self.model.blocks
        for b in list(blocks)[len(blocks) - n :]:
            _set_requires_grad(b, True)
        if hasattr(self.model, "norm"):
            _set_requires_grad(self.model.norm, True)
        logger.info("ViT backbone: training only the last %d/%d blocks (+ final norm)", n, len(blocks))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[list[Tensor], list[Tensor]]:
        H, W = x.shape[-2:]
        pad_h = (self.patch_stride - H % self.patch_stride) % self.patch_stride
        pad_w = (self.patch_stride - W % self.patch_stride) % self.patch_stride
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            if mask is not None:
                mask = F.pad(mask, (0, pad_w, 0, pad_h), value=True)
            else:
                mask = torch.zeros(x.shape[0], H, W, dtype=torch.bool, device=x.device)
                mask = F.pad(mask, (0, pad_w, 0, pad_h), value=True)
        feat = self.model.forward_intermediates(x, indices=[-1], intermediates_only=True, output_fmt="NCHW", norm=True)[0]
        feats = self.pyramid(feat)
        masks = []
        for f in feats:
            if mask is None:
                masks.append(torch.zeros(f.shape[0], f.shape[2], f.shape[3], dtype=torch.bool, device=f.device))
            else:
                masks.append(downsample_mask(mask, (f.shape[2], f.shape[3])))
        return feats, masks


# --------------------------------------------------------------------------- #
def build_backbone(cfg: BackboneConfig) -> nn.Module:
    timm_name = resolve_timm_name(cfg.name)
    kwargs: dict[str, Any] = dict(cfg.timm_kwargs)
    if cfg.drop_path_rate > 0:
        kwargs.setdefault("drop_path_rate", cfg.drop_path_rate)
    # Probe the architecture once (no weights) to decide which adapter to use.
    probe = timm.create_model(timm_name, pretrained=False, num_classes=0, **kwargs)
    if _is_plain_vit(probe):
        del probe
        model = timm.create_model(timm_name, pretrained=cfg.pretrained, num_classes=0, dynamic_img_size=True, **kwargs)
        backbone: nn.Module = TimmViTBackbone(model, timm_name, cfg)
    else:
        del probe
        backbone = TimmFeaturesBackbone(timm_name, cfg)
    logger.info(
        "Backbone %s (%s): channels=%s strides=%s", cfg.name, timm_name, backbone.out_channels, backbone.out_strides
    )
    return backbone
