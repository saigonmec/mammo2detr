"""MammoDETR: multi-task DETR for mammography (image classification + lesion detection)."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config, ImageTargetConfig, ModelConfig
from ..utils.box_ops import inverse_sigmoid
from .backbones import build_backbone
from .denoising import DNMeta, build_dn_queries
from .heads import DetectionHead, ImageClassificationHead, masked_mean
from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer

logger = logging.getLogger(__name__)


class MammoDETR(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        cfg: ModelConfig,
        num_det_classes: int,
        image_targets: list[ImageTargetConfig],
    ):
        super().__init__()
        t = cfg.transformer
        d = t.d_model
        self.cfg = cfg
        self.num_det_classes = num_det_classes
        self.num_queries = t.num_queries
        self.backbone = backbone
        self.num_levels = len(backbone.out_channels)

        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, d, kernel_size=1), nn.GroupNorm(32, d)) for c in backbone.out_channels
        )
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.position_embedding = PositionEmbeddingSine(d // 2, temperature=t.pos_temperature, normalize=True)
        self.transformer = Transformer(t, self.num_levels)

        # Matching queries: content embeddings + learnable 4-D anchors (DAB-DETR)
        self.query_embed = nn.Embedding(t.num_queries, d)
        self.anchor_embed = nn.Embedding(t.num_queries, 4)
        with torch.no_grad():
            anchors = torch.rand(t.num_queries, 4)
            anchors[:, 2:] = anchors[:, 2:] * 0.3 + 0.05  # small-ish initial boxes
            self.anchor_embed.weight.copy_(inverse_sigmoid(anchors))
        self.anchor_embed.weight.requires_grad_(t.learn_anchors)

        # Denoising content encoder (num_classes + 1 "unknown")
        self.label_enc = nn.Embedding(num_det_classes + 1, d)

        # Detection heads (per decoder layer)
        self.det_head = DetectionHead(d, num_det_classes, t.num_decoder_layers)
        if t.box_refine:
            self.transformer.decoder.bbox_embed = self.det_head.bbox_embed

        # Image-level heads
        self.image_cls_mode = cfg.heads.image_cls if image_targets else "none"
        self.image_targets = list(image_targets) if self.image_cls_mode != "none" else []
        self.use_cls_query = self.image_cls_mode in ("query", "hybrid")
        self.use_pool = self.image_cls_mode in ("pool", "hybrid")
        self.num_extra_queries = 1 if self.use_cls_query else 0
        if self.use_cls_query:
            self.cls_query = nn.Embedding(1, d)
        in_dim = d * (int(self.use_cls_query) + int(self.use_pool))
        self.image_heads = nn.ModuleDict(
            {it.name: ImageClassificationHead(in_dim, cfg.heads.cls_hidden_dim, it.num_classes, cfg.heads.cls_dropout) for it in self.image_targets}
        )

    # ------------------------------------------------------------------ #
    @property
    def decoder_layers(self):
        return self.transformer.decoder.layers

    def set_store_attention(self, flag: bool) -> None:
        for layer in self.decoder_layers:
            layer.store_attn = flag

    # ------------------------------------------------------------------ #
    def forward(self, images: Tensor, masks: Tensor | None = None, targets: list[dict[str, Tensor]] | None = None) -> dict:
        feats, feat_masks = self.backbone(images, masks)
        srcs = [proj(f) for proj, f in zip(self.input_proj, feats)]
        pos = [self.position_embedding(m).to(s.dtype) for m, s in zip(feat_masks, srcs)]
        bs = images.shape[0]

        tgt = self.query_embed.weight[None].expand(bs, -1, -1)
        ref = self.anchor_embed.weight.sigmoid()[None].expand(bs, -1, -1)
        refine = torch.ones(self.num_queries, dtype=torch.bool, device=images.device)
        attn_mask = None
        dn_meta: DNMeta | None = None

        if self.training and self.cfg.denoising.enabled and targets is not None:
            dn = build_dn_queries(targets, self.cfg.denoising, self.num_det_classes, self.label_enc, self.num_queries, self.num_extra_queries)
            if dn is not None:
                dn_content, dn_boxes_unsig, attn_mask, dn_meta = dn
                tgt = torch.cat([dn_content.to(tgt.dtype), tgt], 1)
                ref = torch.cat([dn_boxes_unsig.sigmoid(), ref], 1)
                refine = torch.cat([torch.ones(dn_meta.pad_size, dtype=torch.bool, device=images.device), refine])
        if self.use_cls_query:
            tgt = torch.cat([tgt, self.cls_query.weight[None].expand(bs, -1, -1)], 1)
            full = torch.tensor([[0.5, 0.5, 1.0, 1.0]], device=images.device).expand(bs, 1, 4)
            ref = torch.cat([ref, full], 1)
            refine = torch.cat([refine, torch.zeros(1, dtype=torch.bool, device=images.device)])

        tr = self.transformer(srcs, feat_masks, pos, tgt, ref, attn_mask, refine if self.cfg.transformer.box_refine else None)
        hs, refs = tr["hs"], tr["references"]  # [L,B,N,C], [L+1,B,N,4]

        n_det = hs.shape[2] - self.num_extra_queries
        outputs_class, outputs_coord = [], []
        for lid in range(hs.shape[0]):
            h = hs[lid][:, :n_det]
            r = refs[lid][:, :n_det]
            delta = self.det_head.bbox_embed[lid](h)
            coord = (delta + inverse_sigmoid(r)).sigmoid() if self.cfg.transformer.box_refine else (delta + inverse_sigmoid(refs[0][:, :n_det])).sigmoid()
            outputs_class.append(self.det_head.class_embed[lid](h))
            outputs_coord.append(coord)
        outputs_class_t = torch.stack(outputs_class)
        outputs_coord_t = torch.stack(outputs_coord)

        out: dict = {}
        if dn_meta is not None:
            p = dn_meta.pad_size
            dn_class, dn_coord = outputs_class_t[:, :, :p], outputs_coord_t[:, :, :p]
            outputs_class_t, outputs_coord_t = outputs_class_t[:, :, p:], outputs_coord_t[:, :, p:]
            out["dn_outputs"] = {
                "pred_logits": dn_class[-1],
                "pred_boxes": dn_coord[-1],
                "aux_outputs": [{"pred_logits": a, "pred_boxes": b} for a, b in zip(dn_class[:-1], dn_coord[:-1])],
            }
            out["dn_meta"] = dn_meta
        out["pred_logits"] = outputs_class_t[-1]
        out["pred_boxes"] = outputs_coord_t[-1]
        if self.cfg.transformer.aux_loss:
            out["aux_outputs"] = [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class_t[:-1], outputs_coord_t[:-1])]

        # image-level heads
        if self.image_targets:
            feats_img = []
            if self.use_cls_query:
                feats_img.append(hs[-1][:, -1])
            if self.use_pool:
                mem, mmask = tr["memory"], tr["memory_mask"]
                feats_img.append(masked_mean(mem, mmask if mmask.any() else None))
            f = torch.cat(feats_img, -1)
            out["image_feat"] = f
            out["image_logits"] = {name: head(f) for name, head in self.image_heads.items()}
        out["spatial_shapes"] = tr["spatial_shapes"]
        return out


def build_model(cfg: Config) -> MammoDETR:
    backbone = build_backbone(cfg.model.backbone)
    num_det_classes = len(cfg.data.det_class_names)
    model = MammoDETR(backbone, cfg.model, num_det_classes, cfg.data.image_labels)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %.2fM total, %.2fM trainable", n_total / 1e6, n_train / 1e6)
    return model
