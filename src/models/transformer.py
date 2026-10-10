"""Transformer encoder/decoder for :class:`MammoDETR`.

* ``type="standard"``: dense attention over one feature level. The decoder
  uses *conditional* cross-attention (Conditional-DETR / DAB-DETR): content and
  positional terms are decoupled so that a query attends around its anchor box
  from the first iteration - the main fix for slow DETR convergence.
* ``type="deformable"``: multi-scale deformable attention (Deformable-DETR)
  over several feature levels, better for small lesions.

Both decoders use learnable/refined 4-D anchor boxes as positional queries
(DAB-DETR) with iterative box refinement (Deformable-DETR), and accept a
self-attention mask so that denoising queries do not leak into the matching
queries (DN-DETR / DINO).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..config import TransformerConfig
from ..utils.box_ops import inverse_sigmoid
from .attention import ConditionalCrossAttention, MSDeformAttn
from .layers import FFN, MLP, get_clones
from .position_encoding import gen_sineembed_for_position


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class StandardEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, pos: Tensor, key_padding_mask: Tensor | None, **_: object) -> Tensor:
        q = k = src + pos
        src2 = self.self_attn(q, k, src, key_padding_mask=key_padding_mask, need_weights=False)[0]
        src = self.norm1(src + self.dropout1(src2))
        return self.norm2(src + self.dropout2(self.ffn(src)))


class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str, n_levels: int, n_points: int):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, nhead, n_points)
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, pos: Tensor, key_padding_mask: Tensor | None, reference_points: Tensor, spatial_shapes: list[tuple[int, int]], **_: object) -> Tensor:
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        return self.norm2(src + self.dropout2(self.ffn(src)))


class TransformerEncoder(nn.Module):
    def __init__(self, layer: nn.Module, num_layers: int, deformable: bool):
        super().__init__()
        self.layers = get_clones(layer, num_layers)
        self.deformable = deformable

    @staticmethod
    def get_reference_points(spatial_shapes: list[tuple[int, int]], valid_ratios: Tensor, device: torch.device) -> Tensor:
        refs = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            refs.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(refs, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(self, src: Tensor, pos: Tensor, key_padding_mask: Tensor | None, spatial_shapes: list[tuple[int, int]], valid_ratios: Tensor) -> Tensor:
        reference_points = None
        if self.deformable:
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, src.device)
        out = src
        for layer in self.layers:
            out = layer(out, pos, key_padding_mask, reference_points=reference_points, spatial_shapes=spatial_shapes)
        return out


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str, deformable: bool, n_levels: int, n_points: int):
        super().__init__()
        self.deformable = deformable
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        if deformable:
            self.cross_attn: nn.Module = MSDeformAttn(d_model, n_levels, nhead, n_points)
        else:
            self.cross_attn = ConditionalCrossAttention(d_model, nhead, dropout)
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.store_attn = False
        self.last_attn: Tensor | tuple[Tensor, Tensor] | None = None

    def forward(
        self,
        tgt: Tensor,
        query_pos: Tensor,
        query_sine: Tensor,
        memory: Tensor,
        memory_pos: Tensor,
        memory_key_padding_mask: Tensor | None,
        reference_points: Tensor | None,
        spatial_shapes: list[tuple[int, int]],
        self_attn_mask: Tensor | None,
    ) -> Tensor:
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        if self.deformable:
            self.cross_attn.store_sampling = self.store_attn
            tgt2 = self.cross_attn(tgt + query_pos, reference_points, memory, spatial_shapes, memory_key_padding_mask)
            if self.store_attn:
                self.last_attn = self.cross_attn.last_sampling
        else:
            tgt2, attn = self.cross_attn(tgt, query_sine, memory, memory_pos, memory_key_padding_mask, need_weights=self.store_attn)
            if self.store_attn:
                self.last_attn = attn
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        return self.norm3(tgt + self.dropout3(self.ffn(tgt)))


class TransformerDecoder(nn.Module):
    def __init__(self, layer: DecoderLayer, num_layers: int, d_model: int, deformable: bool, temperature: float = 20.0):
        super().__init__()
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.deformable = deformable
        self.temperature = temperature
        self.norm = nn.LayerNorm(d_model)
        # anchor box (4-D sine, 2*d_model dims) -> positional query for self-attention
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        # content-conditioned scaling of the positional part used in cross-attention (DAB-DETR)
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        # Set by the model when iterative box refinement is enabled: ModuleList[num_layers]
        self.bbox_embed: nn.ModuleList | None = None
        self.init_query_scale()

    @torch.no_grad()
    def init_query_scale(self) -> None:
        """Start with an identity scaling of the positional term (learnable afterwards)."""
        last = self.query_scale.layers[-1]
        last.weight.zero_()
        last.bias.fill_(1.0)

    def forward(
        self,
        tgt: Tensor,
        reference_boxes: Tensor,
        memory: Tensor,
        memory_pos: Tensor,
        memory_key_padding_mask: Tensor | None,
        spatial_shapes: list[tuple[int, int]],
        valid_ratios: Tensor,
        self_attn_mask: Tensor | None = None,
        refine_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns ``hs`` [L, B, Nq, C] (normed) and ``references`` [L+1, B, Nq, 4]
        (sigmoid space; ``references[l]`` is the box used as input to layer ``l``)."""
        output = tgt
        refs = [reference_boxes]
        hs = []
        for lid, layer in enumerate(self.layers):
            ref = reference_boxes
            sine_4d = gen_sineembed_for_position(ref, self.d_model // 2, self.temperature)  # [B, Nq, 2*d]
            query_pos = self.ref_point_head(sine_4d)
            query_sine_xy = sine_4d[..., : self.d_model]  # (y, x) part
            if lid > 0:  # Conditional-DETR: first layer uses the raw positional term
                query_sine_xy = query_sine_xy * self.query_scale(output)
            ref_input = None
            if self.deformable:
                ref_input = ref[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            output = layer(output, query_pos, query_sine_xy, memory, memory_pos, memory_key_padding_mask, ref_input, spatial_shapes, self_attn_mask)
            h = self.norm(output)
            hs.append(h)
            if self.bbox_embed is not None:
                delta = self.bbox_embed[lid](h)
                new_ref = (delta + inverse_sigmoid(ref)).sigmoid()
                if refine_mask is not None:
                    new_ref = torch.where(refine_mask[None, :, None], new_ref, ref)
                reference_boxes = new_ref.detach()
            refs.append(reference_boxes)
        return torch.stack(hs), torch.stack(refs)


# --------------------------------------------------------------------------- #
# Full transformer
# --------------------------------------------------------------------------- #
class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig, num_feature_levels: int):
        super().__init__()
        self.cfg = cfg
        self.deformable = cfg.type == "deformable"
        self.d_model = cfg.d_model
        self.num_feature_levels = num_feature_levels
        if self.deformable:
            enc_layer: nn.Module = DeformableEncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout, cfg.activation, num_feature_levels, cfg.enc_n_points)
        else:
            enc_layer = StandardEncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout, cfg.activation)
        self.encoder = TransformerEncoder(enc_layer, cfg.num_encoder_layers, self.deformable) if cfg.num_encoder_layers > 0 else None
        dec_layer = DecoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout, cfg.activation, self.deformable, num_feature_levels, cfg.dec_n_points)
        self.decoder = TransformerDecoder(dec_layer, cfg.num_decoder_layers, cfg.d_model, self.deformable, cfg.pos_temperature)
        self.level_embed = nn.Parameter(torch.zeros(num_feature_levels, cfg.d_model))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
            elif isinstance(m, ConditionalCrossAttention):
                m.init_positional_projections()
            elif isinstance(m, TransformerDecoder):
                m.init_query_scale()
        nn.init.normal_(self.level_embed)

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        _, H, W = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_w.float() / W, valid_h.float() / H], -1)

    def flatten_levels(self, srcs: list[Tensor], masks: list[Tensor], pos_embeds: list[Tensor]):
        src_flatten, mask_flatten, pos_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos) in enumerate(zip(srcs, masks, pos_embeds)):
            b, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_flatten.append(pos.flatten(2).transpose(1, 2) + self.level_embed[lvl].view(1, 1, -1))
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        pos_flatten = torch.cat(pos_flatten, 1)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        return src_flatten, mask_flatten, pos_flatten, spatial_shapes, valid_ratios

    def forward(
        self,
        srcs: list[Tensor],
        masks: list[Tensor],
        pos_embeds: list[Tensor],
        tgt: Tensor,
        reference_boxes: Tensor,
        self_attn_mask: Tensor | None = None,
        refine_mask: Tensor | None = None,
    ) -> dict[str, object]:
        src_flatten, mask_flatten, pos_flatten, spatial_shapes, valid_ratios = self.flatten_levels(srcs, masks, pos_embeds)
        key_padding_mask = mask_flatten if mask_flatten.any() else None
        memory = src_flatten
        if self.encoder is not None:
            memory = self.encoder(src_flatten, pos_flatten, key_padding_mask, spatial_shapes, valid_ratios)
        hs, refs = self.decoder(tgt, reference_boxes, memory, pos_flatten, key_padding_mask, spatial_shapes, valid_ratios, self_attn_mask, refine_mask)
        return {"hs": hs, "references": refs, "memory": memory, "memory_mask": mask_flatten, "spatial_shapes": spatial_shapes}
