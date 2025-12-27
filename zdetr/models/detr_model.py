import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .detr_wrapper import (
    ResNetFeatureWrapper,
    TimmFeatureWrapper,
    DinoFeatureWrapper,
)
from .backbone import get_resnet_backbone, get_timm_backbone, get_dino_backbone
from .backbone import get_mamba_backbone  # ADDED: import get_mamba_backbone

from .detr_model_attn import (
    MultiScaleSpatialAttention,
    PositionEmbeddingSine,
    EfficientDeformableAttention,
)


class EfficientDETRDecoder(nn.Module):
    """
    Efficient DETR Decoder with Denoising Queries:
    - Simple denoising queries from GT boxes
    - IoU-weighted objectness loss
    - Reduced layers for efficiency
    """

    def __init__(self, feature_dim, num_queries=5, num_heads=2, num_layers=1):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.feature_dim = feature_dim

        # Learnable queries and reference points
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        self.ref_point_head = nn.Linear(feature_dim, 2)

        # Box embedding for denoising queries
        self.box_embed = nn.Sequential(
            nn.Linear(4, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim)
        )

        # Positional encoding
        self.pos_encoder = PositionEmbeddingSine(feature_dim // 2)

        # Decoder layers (reduced complexity)
        self.decoder_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(
                            feature_dim, num_heads, dropout=0.1, batch_first=True
                        ),
                        "cross_attn": EfficientDeformableAttention(
                            feature_dim, num_heads, num_points=4
                        ),
                        "ffn": nn.Sequential(
                            nn.Linear(feature_dim, feature_dim * 2),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(feature_dim * 2, feature_dim),
                        ),
                        "norm1": nn.LayerNorm(feature_dim),
                        "norm2": nn.LayerNorm(feature_dim),
                        "norm3": nn.LayerNorm(feature_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # Prediction heads (shared for efficiency)
        self.bbox_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, 4),
                    nn.Sigmoid(),
                )
                for _ in range(num_layers)
            ]
        )

        self.obj_score_head = nn.ModuleList(
            [nn.Linear(feature_dim, 1) for _ in range(num_layers)]
        )

    def generate_noisy_boxes(self, gt_boxes, noise_scale=0.2):
        """
        Generate noisy boxes from GT for denoising queries
        Args:
            gt_boxes: [B, N_gt, 4] normalized boxes
            noise_scale: noise level (default 0.2)
        Returns:
            noisy_boxes: [B, N_gt, 4]
        """
        noise = torch.randn_like(gt_boxes) * noise_scale
        noisy_boxes = (gt_boxes + noise).clamp(0, 1)
        return noisy_boxes

    def forward(self, feat_map, gt_boxes=None, return_attention_maps=False):
        """
        Args:
            feat_map: [B, C, H, W]
            gt_boxes: [B, N_gt, 4] during training, None during inference
            return_attention_maps: bool, whether to return cross-attention maps
        Returns:
            outputs: dict with predictions and global attention map
        """
        B, C, H, W = feat_map.shape

        # Add positional encoding
        pos_embed = self.pos_encoder(feat_map)
        feat_map_with_pos = feat_map + pos_embed
        feat_seq = feat_map_with_pos.flatten(2).permute(0, 2, 1)

        # Initialize learnable queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, N_q, C]
        reference_points = self.ref_point_head(queries).sigmoid()  # [B, N_q, 2]

        # Generate denoising queries if training
        dn_queries = None
        dn_ref_points = None
        num_dn = 0

        if self.training and gt_boxes is not None:
            # Generate noisy boxes
            noisy_boxes = self.generate_noisy_boxes(gt_boxes)
            num_dn = noisy_boxes.shape[1]

            # Encode noisy boxes to queries
            dn_queries = self.box_embed(noisy_boxes)  # [B, N_gt, C]

            # Reference points from noisy box centers
            dn_ref_x = (noisy_boxes[..., 0] + noisy_boxes[..., 2]) / 2
            dn_ref_y = (noisy_boxes[..., 1] + noisy_boxes[..., 3]) / 2
            dn_ref_points = torch.stack([dn_ref_x, dn_ref_y], dim=-1)  # [B, N_gt, 2]

            # Concatenate denoising queries with learnable queries
            queries = torch.cat([dn_queries, queries], dim=1)  # [B, N_dn+N_q, C]
            reference_points = torch.cat([dn_ref_points, reference_points], dim=1)

        all_bbox_preds = []
        all_obj_scores = []
        cross_attn_maps = None

        # Decoder layers
        for layer_idx, layer in enumerate(self.decoder_layers):
            # Self-attention
            queries_norm = layer["norm1"](queries)
            queries2, _ = layer["self_attn"](queries_norm, queries_norm, queries_norm)
            queries = queries + queries2

            # Cross-attention with attention map extraction
            queries_norm = layer["norm2"](queries)

            # Get attention maps from last layer only
            return_attn = return_attention_maps and (layer_idx == self.num_layers - 1)

            if return_attn:
                queries2, attn_maps = layer["cross_attn"](
                    queries_norm,
                    reference_points,
                    feat_seq,
                    (H, W),
                    return_attention_weights=True,
                )
                cross_attn_maps = attn_maps  # [B, H, W] global map
            else:
                # FIXED: cross_attn returns only output when return_attention_weights=False
                queries2 = layer["cross_attn"](
                    queries_norm,
                    reference_points,
                    feat_seq,
                    (H, W),
                    return_attention_weights=False,
                )

            queries = queries + queries2

            # FFN
            queries_norm = layer["norm3"](queries)
            queries2 = layer["ffn"](queries_norm)
            queries = queries + queries2

            # Predict bbox and objectness
            bbox_pred = self.bbox_head[layer_idx](queries)
            obj_score = self.obj_score_head[layer_idx](queries)

            # Iterative refinement
            if layer_idx < self.num_layers - 1:
                new_ref_x = (bbox_pred[..., 0] + bbox_pred[..., 2]) / 2
                new_ref_y = (bbox_pred[..., 1] + bbox_pred[..., 3]) / 2
                reference_points = torch.stack([new_ref_x, new_ref_y], dim=-1).detach()

            all_bbox_preds.append(bbox_pred)
            all_obj_scores.append(obj_score)

        # Split denoising and matching outputs
        if num_dn > 0:
            dn_bbox_preds = [pred[:, :num_dn] for pred in all_bbox_preds]
            dn_obj_scores = [score[:, :num_dn] for score in all_obj_scores]

            match_bbox_preds = [pred[:, num_dn:] for pred in all_bbox_preds]
            match_obj_scores = [score[:, num_dn:] for score in all_obj_scores]

            return {
                "pred_bboxes": match_bbox_preds[-1],
                "obj_scores": match_obj_scores[-1],
                "aux_outputs": [
                    {"pred_bboxes": bbox, "obj_scores": score}
                    for bbox, score in zip(match_bbox_preds[:-1], match_obj_scores[:-1])
                ],
                "dn_outputs": [
                    {"pred_bboxes": bbox, "obj_scores": score, "gt_boxes": gt_boxes}
                    for bbox, score in zip(dn_bbox_preds, dn_obj_scores)
                ],
                "cross_attn_maps": cross_attn_maps,  # [B, H, W]
            }
        else:
            return {
                "pred_bboxes": all_bbox_preds[-1],
                "obj_scores": all_obj_scores[-1],
                "aux_outputs": [
                    {"pred_bboxes": bbox, "obj_scores": score}
                    for bbox, score in zip(all_bbox_preds[:-1], all_obj_scores[:-1])
                ],
                "dn_outputs": [],
                "cross_attn_maps": cross_attn_maps,  # [B, H, W]
            }


class M2DETRModel(nn.Module):
    """
    Efficient Multi-task DETR for Mammography
    """

    def __init__(
        self, backbone, feature_dim, num_classes=2, num_queries=5, reduced_dim=256
    ):
        super().__init__()
        self.backbone = backbone
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if feature_dim > reduced_dim:
            self.feature_proj = nn.Conv2d(feature_dim, reduced_dim, kernel_size=1)
            feature_dim = reduced_dim
        else:
            self.feature_proj = None

        self.classifier = nn.Linear(feature_dim, num_classes)

        # UPDATED: Use improved multi-scale spatial attention
        self.spatial_attention = MultiScaleSpatialAttention(feature_dim, reduction=8)

        self.cls_token_proj = nn.Linear(
            backbone.base_model.embed_dim
            if hasattr(backbone, "base_model")
            and hasattr(backbone.base_model, "embed_dim")
            else feature_dim,
            feature_dim,
        )

        self.detr_decoder = EfficientDETRDecoder(
            feature_dim,
            num_queries=num_queries,
            num_heads=1,
            num_layers=1,
        )

    def forward(self, x, gt_boxes=None, return_attention_maps=False):
        """
        Args:
            x: input images
            gt_boxes: [B, N_gt, 4] ground truth boxes (training only)
            return_attention_maps: bool, whether to return decoder attention maps
        Returns:
            dict with cls_logits, pred_bboxes, obj_scores, attn_maps [B, H, W]
        """
        # Backbone
        feat_map = self.backbone(x)

        # Nếu là dict (ViT/DINO), lấy cls và spatial
        if isinstance(feat_map, dict):
            feat_map_dict = feat_map  # keep original dict
            feat_map = feat_map_dict["spatial"]
            cls_token = feat_map_dict.get("cls", None)
        else:
            feat_map = feat_map
            cls_token = None

        # Ensure 4D
        if feat_map.dim() == 2:
            feat_map = feat_map.unsqueeze(-1).unsqueeze(-1)
        elif feat_map.dim() != 4:
            raise ValueError(f"Unexpected backbone shape: {feat_map.shape}")

        # ADDED: Project to lower dimension
        if self.feature_proj is not None:
            feat_map = self.feature_proj(feat_map)

        # Classification: ưu tiên dùng CLS token nếu có, không thì dùng pooled spatial
        if cls_token is not None:
            # cls_output = self.classifier(cls_token)
            if cls_token.shape[-1] != feat_map.shape[1]:
                cls_token = self.cls_token_proj(cls_token)
            cls_output = self.classifier(cls_token)

        else:
            cls_feat = self.global_pool(feat_map).flatten(1)
            cls_output = self.classifier(cls_feat)

        # cls_feat = self.global_pool(feat_map).flatten(1)
        # cls_output = self.classifier(cls_feat)

        # Get spatial attention (keep for feature enhancement)
        attn_feat, spatial_attn_map = self.spatial_attention(feat_map)

        # Get decoder outputs with optional cross-attention maps
        detr_outputs = self.detr_decoder(attn_feat, gt_boxes, return_attention_maps)

        return {
            "cls_logits": cls_output,
            "pred_bboxes": detr_outputs["pred_bboxes"],
            "obj_scores": detr_outputs["obj_scores"],
            "aux_outputs": detr_outputs["aux_outputs"],
            "dn_outputs": detr_outputs.get("dn_outputs", []),
            "attn_maps": detr_outputs.get(
                "cross_attn_maps"
            ),  # UPDATED: [B, H, W] global attention map
            "spatial_attn": spatial_attn_map,  # [B, 1, H, W]
        }


def get_detr_model(
    model_type="resnet50", num_classes=2, dino_unfreeze_blocks=2, num_queries=5
):
    """Get efficient M2 DETR model (num_queries=5 for mammography)"""
    if model_type in ["resnet34", "resnet50", "resnet101", "resnext50", "resnet152"]:
        backbone, feature_dim = get_resnet_backbone(model_type)
        backbone = ResNetFeatureWrapper(backbone)
    elif model_type in [
        "resnest50",
        "resnest101",
        "resnest50s2",
        "regnety",
        "convnextv2base",
        "convnextv2_tiny",
        "efficientnetv2",
        "efficientnetv2s",
        "maxvit_tiny",
        "maxvit_tiny512",
        "maxvit_small",
        "maxvit_base",
        "eva02_small",
        "eva02_base",
        "vit_small",
        "swinv2_tiny",
        "swinv2_base",
        "swinv2_small",
        "mambaout_tiny",
    ]:
        backbone, feature_dim = get_timm_backbone(model_type)
        if hasattr(backbone, "forward_features"):
            backbone = TimmFeatureWrapper(backbone)

    elif model_type in [
        "dinov2_small",
        "dinov2_base",
        "dinov3_convnext_tiny",
        "dinov3_convnext_small",
        "dinov3_vit16small",
        "dinov3_vit16smallplus",
        "dinov3_vit16base",
        "dinov3_vit16large",
        "dinov3_convnext_base",
        "dinov3_convnext_large",
        "medino_vitb16",
        "dinov2uni_base",
    ]:
        backbone, feature_dim = get_dino_backbone(model_type)
        # Ensure patch_size=16 for ViT-16 models
        backbone = DinoFeatureWrapper(backbone)
        # Unfreeze last blocks for dino/vit models
        unfreeze_last_blocks(
            backbone.base_model if hasattr(backbone, "base_model") else backbone,
            dino_unfreeze_blocks,
        )
    elif model_type in ["mamba_t", "mamba_s"]:  # ADDED: mamba support
        backbone, feature_dim = get_mamba_backbone(model_type)
        # No wrapper needed for mamba backbone
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = M2DETRModel(
        backbone, feature_dim, num_classes, num_queries, reduced_dim=256
    )
    return model


def unfreeze_last_blocks(model, num_blocks=2):
    """
    Unfreeze the last `num_blocks` transformer blocks of a ViT/DINO backbone,
    hoặc các block cuối của ConvNeXt/DINOv3 (có thuộc tính stages).
    Print thông tin đơn giản về các block được unfreeze.
    """
    print(f"[INFO] Unfreezing last {num_blocks} blocks of the model...")

    # --- ViT-style (blocks/layers/transformer.blocks) ---
    block_attrs = ["blocks", "layers", "transformer.blocks"]
    for attr in block_attrs:
        blocks = None
        obj = model
        for part in attr.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                obj = None
                break
        blocks = obj
        if blocks is not None and hasattr(blocks, "__getitem__"):
            total_blocks = len(blocks)
            unfrozen_layers = []
            for i in range(total_blocks - num_blocks, total_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
                unfrozen_layers.append(i)
            # Freeze all other blocks
            for i in range(0, total_blocks - num_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = False
            print(f"[INFO] Unfroze {len(unfrozen_layers)} layers: {unfrozen_layers}")
            for i in unfrozen_layers:
                print(f"  - Layer {i}: {blocks[i].__class__.__name__}")
                for name, module in blocks[i].named_children():
                    print(f"    - Submodule: {name} ({module.__class__.__name__})")
            return  # Done

    # --- ConvNeXt/DINOv3-style (stages) ---
    if hasattr(model, "stages"):
        stages = model.stages
        for stage_idx, stage in enumerate(stages):
            if hasattr(stage, "blocks"):
                blocks = stage.blocks
                total_blocks = len(blocks)
                unfrozen_layers = []
                for i in range(total_blocks - num_blocks, total_blocks):
                    for param in blocks[i].parameters():
                        param.requires_grad = True
                    unfrozen_layers.append(i)
                # Freeze all other blocks
                for i in range(0, total_blocks - num_blocks):
                    for param in blocks[i].parameters():
                        param.requires_grad = False
                print(
                    f"[INFO] Stage {stage_idx}: Unfroze {len(unfrozen_layers)} blocks: {unfrozen_layers}"
                )
            else:
                print(f"[WARN] Stage {stage_idx} has no 'blocks' attribute.")
        # Optionally, unfreeze stem if needed
        if hasattr(model, "stem"):
            for module in model.stem.modules():
                for param in getattr(module, "parameters", lambda: [])():
                    param.requires_grad = True
            print("[INFO] Unfroze stem layers.")
        return

    print("[WARN] No recognized block structure found for unfreezing.")
