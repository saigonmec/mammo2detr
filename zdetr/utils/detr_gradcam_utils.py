"""
GradCAM utilities for DETR models
- Layer detection for different backbones
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from numpy import ndarray


def vit_reshape_transform(x, grid_h, grid_w):
    """
    Reshape ViT output từ [B, N, C] sang [B, C, H, W]
    Loại bỏ CLS token và extra tokens
    """
    B, N, C = x.shape
    num_patches = grid_h * grid_w

    # Tính số extra tokens (CLS + register tokens)
    num_extra = N - num_patches

    if num_extra < 0:
        raise ValueError(
            f"Mismatch: Expected {num_patches} patches but got {N} tokens. "
            f"Grid size ({grid_h}, {grid_w}) may be incorrect."
        )

    # Loại bỏ CLS token và register tokens (thường ở đầu)
    x = x[:, num_extra:, :]  # [B, num_patches, C]

    if x.shape[1] != num_patches:
        raise ValueError(
            f"After removing {num_extra} extra tokens, got {x.shape[1]} tokens "
            f"but expected {num_patches} patches."
        )

    # Reshape thành spatial grid
    x = x.reshape(B, grid_h, grid_w, C)  # [B, H, W, C]
    return x.permute(0, 3, 1, 2)  # [B, C, H, W]


def _detect_vit_grid_size(model, input_h, input_w):
    """
    Detect if model is ViT and calculate grid size from input dimensions

    Returns:
        tuple: (is_vit, grid_h, grid_w)
    """

    # Helper function to extract patch_size
    def get_patch_size(patch_embed):
        if not hasattr(patch_embed, "patch_size"):
            return None
        ps = patch_embed.patch_size
        return ps[0] if hasattr(ps, "__getitem__") else ps

    # Helper function to get grid from patch_embed
    def get_grid_from_patch_embed(patch_embed):
        patch_size = get_patch_size(patch_embed)
        if patch_size is not None:
            return True, input_h // patch_size, input_w // patch_size

        # Fallback: use grid_size attribute
        if hasattr(patch_embed, "grid_size"):
            return True, patch_embed.grid_size[0], patch_embed.grid_size[1]

        # Fallback: calculate from num_patches
        if hasattr(patch_embed, "num_patches"):
            num_patches = patch_embed.num_patches
            grid = int(num_patches**0.5)
            return True, grid, grid

        return False, None, None

    # --- MambaVision detection ---
    # MambaVision có downsample ở mỗi level: input/4, input/8, input/16, input/32
    # Level 3 (cuối) thường là input/32
    if hasattr(model, "backbone") and hasattr(model.backbone, "base_model"):
        if hasattr(model.backbone.base_model, "model") and hasattr(
            model.backbone.base_model.model, "levels"
        ):
            # Đây là MambaVision structure
            # Level 0: /4, Level 1: /8, Level 2: /16, Level 3: /32
            grid_h = input_h // 32
            grid_w = input_w // 32
            print(
                f"[DEBUG] Detected MambaVision: grid_size = {grid_h}x{grid_w} (input: {input_h}x{input_w})"
            )
            return True, grid_h, grid_w

    # Case 1: model.transformer.patch_embed (DinoVisionTransformerClassifier wrapper)
    if hasattr(model, "transformer") and hasattr(model.transformer, "patch_embed"):
        return get_grid_from_patch_embed(model.transformer.patch_embed)

    # Case 2: model.backbone.base_model.patch_embed (DETR with ViT backbone)
    if hasattr(model, "backbone") and hasattr(model.backbone, "base_model"):
        if hasattr(model.backbone.base_model, "patch_embed"):
            return get_grid_from_patch_embed(model.backbone.base_model.patch_embed)

    # Case 3: model.patch_embed (direct ViT model)
    if hasattr(model, "patch_embed"):
        return get_grid_from_patch_embed(model.patch_embed)

    return False, None, None


def _process_vit_activations(acts, grads, model, input_shape):
    """
    Process ViT activations: detect grid size and reshape if needed

    Returns:
        tuple: (acts, grads) - reshaped to [B, C, H, W] if ViT, unchanged otherwise
    """
    # Skip if already 4D (CNN activations)
    if acts.ndim == 4:
        return acts, grads

    # Only process 3D tensors (ViT format [B, N, C])
    if acts.ndim != 3:
        return acts, grads

    # Detect ViT and get grid size
    input_h, input_w = input_shape[2], input_shape[3]
    is_vit, grid_h, grid_w = _detect_vit_grid_size(model, input_h, input_w)

    if not is_vit:
        return acts, grads

    if grid_h is None or grid_w is None:
        raise RuntimeError(
            f"ViT model detected but grid_size is None. Cannot reshape activations.\n"
            f"Acts shape: {acts.shape}, Input shape: {input_shape}"
        )

    # Reshape ViT activations
    try:
        acts = vit_reshape_transform(acts, grid_h, grid_w)
        grads = vit_reshape_transform(grads, grid_h, grid_w)
        # print(f"[DEBUG] ViT reshape: {acts.shape} (grid: {grid_h}x{grid_w})")
        return acts, grads
    except ValueError as e:
        raise RuntimeError(
            f"Failed to reshape ViT activations: {e}\n"
            f"Input: {input_shape}, Acts: {acts.shape}, Grid: ({grid_h}, {grid_w})"
        )


def gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str | nn.Module,
    class_idx: int | None = None,
) -> ndarray:
    """
    GradCAM for both CNN and ViT. Auto-detects model type and reshapes accordingly.
    """
    # Setup model for gradient computation
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    # Prepare hooks
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Get target layer
    if isinstance(target_layer, str):
        try:
            layer = dict([*model.named_modules()])[target_layer]
        except KeyError:
            available = list(dict([*model.named_modules()]).keys())
            raise ValueError(
                f"Layer '{target_layer}' not found. Available: {available}"
            )
    else:
        layer = target_layer

    # Register hooks
    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        with torch.set_grad_enabled(True):
            output = model(input_tensor)

            # Handle DETR output (dict with 'cls_logits')
            if isinstance(output, dict):
                if "cls_logits" not in output:
                    raise ValueError(
                        f"Model returned dict without 'cls_logits'. Keys: {list(output.keys())}"
                    )
                output = output["cls_logits"]

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            # Backward pass
            model.zero_grad()
            output[0, class_idx].backward()

        # Check hooks captured data
        if len(activations) == 0 or len(gradients) == 0:
            raise RuntimeError(
                f"Hooks failed for '{target_layer}'. "
                f"Acts: {len(activations)}, Grads: {len(gradients)}"
            )

        acts = activations[0]
        grads = gradients[0]

        # print(f"[DEBUG] Raw shapes - acts: {acts.shape}, grads: {grads.shape}")

        # Process ViT activations if needed (reshape [B, N, C] -> [B, C, H, W])
        acts, grads = _process_vit_activations(acts, grads, model, input_tensor.shape)

        # Validate 4D tensors
        if acts.ndim != 4 or grads.ndim != 4:
            raise RuntimeError(
                f"Expected 4D tensors [B,C,H,W], got acts: {acts.shape}, grads: {grads.shape}"
            )

        # Compute GradCAM
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.uint8(cam * 255)

        return cam

    finally:
        # Always cleanup hooks
        handle_f.remove()
        handle_b.remove()


def _get_mamba_gradcam_layer(model, model_name, has_backbone):
    """
    Tìm layer phù hợp cho GradCAM trong MambaVision models.
    Ưu tiên các layer có output 4D (conv blocks) để tránh vấn đề reshape.

    Args:
        model: The model instance
        model_name: Model architecture name (e.g., 'mamba_t', 'mamba_s')
        has_backbone: Whether model has backbone wrapper

    Returns:
        str: Layer name for GradCAM
    """
    prefix = "backbone." if has_backbone else ""

    print(f"\n{'=' * 60}")
    print(f"DEBUG: MambaVision GradCAM Layer Detection")
    print(f"Model: {model_name}")
    print(f"Has backbone wrapper: {has_backbone}")
    print(f"{'=' * 60}\n")

    named_modules = dict([*model.named_modules()])

    # Strategy 1: Downsample layers (output 4D: [B, C, H, W])
    print("[Strategy 1] Looking for downsample layers...")
    downsample_layers = [
        k for k in named_modules if "downsample" in k and "reduction" in k
    ]
    if downsample_layers:
        # Lấy downsample cuối cùng (level 2 -> level 3)
        selected = downsample_layers[-1]
        print(f"✓ Found downsample layer: {selected}")
        print(f"  Output shape: 4D [B, C, H, W] (no reshape needed)")
        print(f"{'=' * 60}\n")
        return f"{prefix}{selected}" if not selected.startswith(prefix) else selected

    # Strategy 2: Conv blocks trong level 0, 1 (output 4D)
    print("[Strategy 2] Looking for conv blocks in level 0, 1...")
    conv_blocks = [
        k
        for k in named_modules
        if ("levels.0.blocks" in k or "levels.1.blocks" in k) and k.endswith("norm2")
    ]
    if conv_blocks:
        selected = sorted(conv_blocks)[-1]
        print(f"✓ Found conv block: {selected}")
        print(f"  Output shape: 4D [B, C, H, W] (no reshape needed)")
        print(f"{'=' * 60}\n")
        return f"{prefix}{selected}" if not selected.startswith(prefix) else selected

    # Strategy 3: Last conv layer in level 1
    print("[Strategy 3] Looking for last conv layer in level 1...")
    level1_convs = [k for k in named_modules if "levels.1.blocks" in k and "conv" in k]
    if level1_convs:
        selected = sorted(level1_convs)[-1]
        print(f"✓ Found level 1 conv: {selected}")
        print(f"  Output shape: 4D [B, C, H, W] (no reshape needed)")
        print(f"{'=' * 60}\n")
        return f"{prefix}{selected}" if not selected.startswith(prefix) else selected

    # Strategy 4: Mamba/Attention blocks (output 3D, cần reshape)
    print("[Strategy 4] Looking for Mamba/Attention blocks (requires reshape)...")
    mamba_blocks = [
        k for k in named_modules if "levels" in k and "blocks" in k and "norm1" in k
    ]
    if mamba_blocks:
        # Lấy block cuối của level cao nhất
        def extract_level_block(name):
            import re

            level_match = re.search(r"levels\.(\d+)", name)
            block_match = re.search(r"blocks\.(\d+)", name)
            level = int(level_match.group(1)) if level_match else -1
            block = int(block_match.group(1)) if block_match else -1
            return (level, block)

        sorted_blocks = sorted(mamba_blocks, key=extract_level_block, reverse=True)
        selected = sorted_blocks[0]
        print(f"⚠ Found Mamba/Attention block: {selected}")
        print(f"  Output shape: 3D [B, N, C] (will be reshaped to 4D)")
        print(f"  Note: Requires grid_size detection for reshape")
        print(f"{'=' * 60}\n")
        return f"{prefix}{selected}" if not selected.startswith(prefix) else selected

    # Fallback: feature_proj
    print(f"[Fallback] Using feature_proj")
    print(f"{'=' * 60}\n")
    return f"{prefix}feature_proj"


def get_gradcam_layer(model, model_name):
    """
    Return the appropriate layer name for GradCAM based on model_name.
    Automatically detects if model has backbone wrapper and adds prefix.

    Args:
        model: The model instance
        model_name: Model architecture name (e.g., 'resnet50', 'dinov2_small')

    Returns:
        str: Layer name for GradCAM (e.g., 'backbone.layer4' for DETR models)
    """
    # Check if model has backbone (DETR models)
    has_backbone = hasattr(model, "backbone")
    prefix = "backbone." if has_backbone else ""

    def find_deepest_layer(model, candidates):
        """Find the deepest valid layer from candidates"""
        named_modules = dict([*model.named_modules()])
        for cand in candidates:
            for name in named_modules:
                if name.endswith(cand):
                    return name
        return None

    # MambaVision models
    if "mamba" in model_name:
        return _get_mamba_gradcam_layer(model, model_name, has_backbone)

    # ResNet, ResNeXt, ResNeSt
    if "resnet" in model_name or "resnext" in model_name or "resnest" in model_name:
        return f"{prefix}layer4"

    # ConvNeXt
    elif "convnext" in model_name:
        candidates = [
            "base_model.stages.3.blocks.2.conv_dw",
            "stages.3.blocks.2.conv_dw",
            "stages.3.blocks.2",
        ]
        layer_name = find_deepest_layer(model, candidates)
        if layer_name:
            return (
                f"{prefix}{layer_name}"
                if not layer_name.startswith(prefix)
                else layer_name
            )
        return f"{prefix}feature_proj"

    # MaxViT
    elif "maxvit" in model_name:
        candidates = [
            "base_model.stages.3.blocks.1.conv.norm2",
            "base_model.stages.3.blocks.1.conv",
            "base_model.stages.3.blocks.1",
        ]
        layer_name = find_deepest_layer(model, candidates)
        if layer_name:
            return (
                f"{prefix}{layer_name}"
                if not layer_name.startswith(prefix)
                else layer_name
            )
        return f"{prefix}feature_proj"

    # RegNetY
    elif "regnety" in model_name:
        candidates = [
            "base_model.s4.b1.conv3.conv",
            "s4.b1.conv3.conv",
        ]
        layer_name = find_deepest_layer(model, candidates)
        if layer_name:
            return (
                f"{prefix}{layer_name}"
                if not layer_name.startswith(prefix)
                else layer_name
            )
        return f"{prefix}feature_proj"

    # EfficientNet
    elif "efficientnet" in model_name:
        candidates = [
            "base_model.blocks.5.2.conv_pwl",
            "base_model.blocks.5.2",
            "blocks.5.2.conv_pwl",
            "blocks.5.2",
        ]
        layer_name = find_deepest_layer(model, candidates)
        if layer_name:
            return (
                f"{prefix}{layer_name}"
                if not layer_name.startswith(prefix)
                else layer_name
            )
        return f"{prefix}feature_proj"

    # DINOv2 and DINOv3
    # elif "dinov2" in model_name or "dinov3" in model_name:
    #    return f"{prefix}transformer.blocks.23.norm1"

    elif "dinov2" in model_name or "dinov3" in model_name or "vit" in model_name:
        # Tìm block cuối cùng của backbone.base_model.blocks
        named_modules = dict([*model.named_modules()])
        block_names = [
            k for k in named_modules if k.endswith("norm1") and "blocks." in k
        ]
        if block_names:
            # Lấy block.norm1 có số lớn nhất
            last_block = sorted(
                block_names, key=lambda x: int(x.split("blocks.")[1].split(".")[0])
            )[-1]
            return last_block
        # Fallback
        return f"{prefix}base_model.blocks.0.norm1"

    # Swin Transformer
    elif "swin" in model_name:
        return f"{prefix}layers.-1.blocks.-1.norm1"

    # EVA02
    elif "eva02" in model_name:
        return f"{prefix}blocks.-1.norm1"

    # Fallback: try to get last layer
    else:
        children = list(model.named_children())
        if children:
            last_child_name = children[-1][0]
            if last_child_name == "backbone" and has_backbone:
                backbone = children[-1][1]
                backbone_children = list(backbone.named_children())
                if backbone_children:
                    return f"backbone.{backbone_children[-1][0]}"
            return last_child_name
        return None
