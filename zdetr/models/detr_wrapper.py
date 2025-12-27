import torch
import torch.nn as nn
from .backbone import (
    get_resnet_backbone,
    get_timm_backbone,
    get_dino_backbone,
    get_mamba_backbone,
)


class SpatialAttention(nn.Module):
    """Spatial attention on feature map"""

    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W] -> attn_map: [B, 1, H, W]
        attn_map = self.attention(x)
        return x * attn_map, attn_map


# ============= Feature Wrappers (moved outside M2Model) =============


class ResNetFeatureWrapper(nn.Module):
    """Wrapper for ResNet to extract feature map before avgpool"""

    def __init__(self, base_model):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, C, H, W]


class TimmFeatureWrapper(nn.Module):
    """Wrapper for timm models to extract spatial features"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        feat = self.base_model.forward_features(x)
        # Handle different output formats
        if feat.ndim == 4 and feat.shape[-1] < feat.shape[1]:
            return feat  # Already [B, C, H, W]
        elif feat.ndim == 4:
            return feat.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        elif feat.ndim == 3:
            # ViT: [B, N, C] -> [B, C, H, W]
            B, N, C = feat.shape
            H = W = int(N**0.5)
            return feat.transpose(1, 2).reshape(B, C, H, W)
        return feat


class DinoFeatureWrapper_working(nn.Module):
    """Wrapper for Dino/ViT models to extract spatial features only"""

    def __init__(self, base_model, patch_size=16):
        super().__init__()
        self.base_model = base_model
        self.patch_size = patch_size

    def forward(self, x):
        print(f"[DEBUG] Input shape: {x.shape}")
        _, _, H, W = x.shape
        pad_H = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_W = (self.patch_size - (W % self.patch_size)) % self.patch_size
        if pad_H > 0 or pad_W > 0:
            print(f"[DEBUG] Padding input: pad_H={pad_H}, pad_W={pad_W}")
            x = nn.functional.pad(x, (0, pad_W, 0, pad_H), mode="constant", value=0)

        # Update H, W after padding
        _, _, H, W = x.shape

        # Lấy đặc trưng từ backbone
        if hasattr(self.base_model, "forward_features"):
            feat = self.base_model.forward_features(x)
        else:
            feat = self.base_model(x)

        print(f"[DEBUG] Feature shape after backbone: {feat.shape}")

        # Nếu là [B, N+1, C] (ViT/DINO), tách CLS và patch tokens
        if feat.ndim == 3:
            # Tính số patch thực sự dựa trên input size
            num_patches = (H // self.patch_size) * (W // self.patch_size)
            print(f"[DEBUG] Expected num_patches: {num_patches}")

            # Lấy đúng số patch tokens (bỏ qua CLS token ở index 0)
            patch_tokens = feat[:, 1 : 1 + num_patches, :]
            B, N, C = patch_tokens.shape
            H_grid = W_grid = int(N**0.5)

            print(
                f"[DEBUG] Patch tokens: B={B}, N={N}, C={C}, H_grid={H_grid}, W_grid={W_grid}"
            )

            if H_grid * W_grid != N:
                print(
                    f"[ERROR] Cannot reshape: num_patches={N} is not a perfect square. Check input size and patch size."
                )
                raise ValueError(
                    f"Cannot reshape: num_patches={N} is not a perfect square. "
                    f"Check input size and patch size."
                )
            spatial_feat = patch_tokens.transpose(1, 2).reshape(B, C, H_grid, W_grid)
            print(f"[DEBUG] Spatial feature shape: {spatial_feat.shape}")
            return spatial_feat  # Return [B, C, H, W] tensor
        # Nếu là [B, C, H, W] (ConvNeXt, ...), không có CLS
        elif feat.ndim == 4:
            print(f"[DEBUG] Feature is already spatial: {feat.shape}")
            return feat  # Return [B, C, H, W] tensor
        else:
            print(f"[ERROR] Unexpected feature shape: {feat.shape}")
            raise ValueError(f"Unexpected feature shape: {feat.shape}")


class DinoFeatureWrapper(nn.Module):
    """Wrapper for Dino/ViT models to extract both CLS token and spatial features"""

    def __init__(self, base_model, patch_size=16):
        super().__init__()
        self.base_model = base_model
        self.patch_size = patch_size

    def forward(self, x):
        # --- PATCH: Pad input so H, W chia hết cho patch_size ---
        _, _, H, W = x.shape
        pad_H = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_W = (self.patch_size - (W % self.patch_size)) % self.patch_size
        if pad_H > 0 or pad_W > 0:
            x = nn.functional.pad(x, (0, pad_W, 0, pad_H), mode="constant", value=0)
        # --- END PATCH ---

        # Update H, W after padding
        _, _, H, W = x.shape

        # Lấy đặc trưng từ backbone
        if hasattr(self.base_model, "forward_features"):
            feat = self.base_model.forward_features(x)
        else:
            feat = self.base_model(x)

        # Nếu là [B, N+1, C] (ViT/DINO), tách CLS và patch tokens
        if feat.ndim == 3:
            num_patches = (H // self.patch_size) * (W // self.patch_size)
            patch_tokens = feat[:, 1 : 1 + num_patches, :]
            B, N, C = patch_tokens.shape

            # Tìm H_grid, W_grid sao cho H_grid * W_grid = N
            H_grid = H // self.patch_size
            W_grid = W // self.patch_size
            if H_grid * W_grid != N:
                print(f"[ERROR] H_grid={H_grid}, W_grid={W_grid}, N={N}")
                raise ValueError(
                    f"Cannot reshape: num_patches={N} != H_grid*W_grid={H_grid * W_grid}. "
                    f"Check input size and patch size."
                )
            spatial_feat = patch_tokens.transpose(1, 2).reshape(B, C, H_grid, W_grid)
            return {"cls": feat[:, 0, :], "spatial": spatial_feat}

        # Nếu là [B, C, H, W] (ConvNeXt, ...), không có CLS
        elif feat.ndim == 4:
            return {"cls": None, "spatial": feat}
        else:
            raise ValueError(f"Unexpected feature shape: {feat.shape}")


class DinoFeatureWrapper_ori(nn.Module):
    """Wrapper for Dino/ViT models to extract spatial features"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        if hasattr(self.base_model, "forward_features"):
            feat = self.base_model.forward_features(x)
        else:
            feat = self.base_model(x)

        # [B, N+1, C] -> [B, C, H, W] (remove CLS token)
        if feat.ndim == 3:
            B, N, C = feat.shape
            feat = feat[:, 1:, :]  # Remove CLS
            H = W = int((N - 1) ** 0.5)
            return feat.transpose(1, 2).reshape(B, C, H, W)
        return feat


# ============= M2Model (simplified, no nested wrappers) =============


class M2Model(nn.Module):
    """Multi-task model: classification + bbox regression"""

    def __init__(self, backbone, feature_dim, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Spatial attention for bbox
        self.spatial_attention = SpatialAttention(feature_dim)

        # BBox regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Backbone feature map
        feat_map = self.backbone(x)  # Should be [B, C, H, W]

        # Ensure feature map is 4D
        if feat_map.dim() == 2:
            feat_map = feat_map.unsqueeze(-1).unsqueeze(-1)
        elif feat_map.dim() != 4:
            raise ValueError(
                f"Unexpected backbone output shape: {feat_map.shape}. Expected 4D [B, C, H, W]"
            )

        # Classification branch (no attention)
        cls_feat = self.global_pool(feat_map).flatten(1)  # [B, C]
        cls_output = self.classifier(cls_feat)

        # BBox branch with attention
        attn_feat, attn_map = self.spatial_attention(feat_map)  # [B, C, H, W]
        bbox_feat = self.global_pool(attn_feat).flatten(1)  # [B, C]
        bbox_output = self.bbox_head(bbox_feat)

        return cls_output, bbox_output, attn_map


def unfreeze_last_blocks(model, num_blocks=2):
    """
    Unfreeze the last `num_blocks` transformer blocks of a ViT/DINO backbone.
    Print how many layers are unfrozen, which layers, and their submodules.
    """
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
            # Print info
            print(f"[INFO] Unfroze {len(unfrozen_layers)} layers: {unfrozen_layers}")
            for i in unfrozen_layers:
                print(f"  - Layer {i}: {blocks[i].__class__.__name__}")
                for name, module in blocks[i].named_children():
                    print(f"    - Submodule: {name} ({module.__class__.__name__})")
            return  # Done
    # If not found, do nothing


def get_m2_model(model_type="resnet50", num_classes=2, dino_unfreeze_blocks=2):
    """Get multi-task model based on model_type"""
    # Get backbone
    if model_type in ["resnet34", "resnet50", "resnet101", "resnext50", "resnet152"]:
        backbone, feature_dim = get_resnet_backbone(model_type)
        backbone = ResNetFeatureWrapper(backbone)

    elif model_type in ["mamba_t", "mamba_s"]:
        raise ValueError(
            f"Mamba models don't support spatial feature maps for M2. Use CNN-based models."
        )

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
        else:
            # Fallback: remove heads
            if hasattr(backbone, "fc"):
                backbone.fc = nn.Identity()
            elif hasattr(backbone, "head") and hasattr(backbone.head, "fc"):
                backbone.head.fc = nn.Identity()
            elif hasattr(backbone, "head"):
                backbone.head = nn.Identity()
            elif hasattr(backbone, "classifier"):
                backbone.classifier = nn.Identity()

    elif model_type in [
        "dinov2_small",
        "dinov2_base",
        "dinov2_small_reg",
        "dinov2_base_reg",
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
        backbone = DinoFeatureWrapper(backbone)
        # Unfreeze last blocks for dino/vit models
        unfreeze_last_blocks(
            backbone.base_model if hasattr(backbone, "base_model") else backbone,
            dino_unfreeze_blocks,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Create M2 model
    model = M2Model(backbone, feature_dim, num_classes)
    return model


# Danh sách model cần test
model_configs = {
    "resnet50": 448,
    "resnest50": 448,
    "convnextv2_tiny": 448,
    "efficientnetv2s": 448,
    "maxvit_tiny": 448,
    "eva02_small": 336,  # Hỗ trợ dynamic
    "vit_small": 224,  # Hỗ trợ dynamic
    "swinv2_tiny": 256,  # Yêu cầu 256
    "dinov2_small": 518,  # Yêu cầu 224 (hoặc 518)
    "dinov3_vit16small": 224,
}

# for model_type, img_size in model_configs.items():
#    try:
#        print(f"\nTesting {model_type} with input {img_size}x{img_size}...")
#        model = get_m2_model(model_type=model_type, num_classes=2)
#        dummy_input = torch.randn(2, 3, img_size, img_size)
#        cls_out, bbox_out, attn_map = model(dummy_input)
#        print(
#            f"  ✅ cls: {cls_out.shape}, bbox: {bbox_out.shape}, attn: {attn_map.shape}"
#        )
#    except Exception as e:
#        print(f"  ❌ Error: {e}")
