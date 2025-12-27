"""
DETR attention map visualization utilities
"""

import torch
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu


def prepare_attention_heatmap_ori(attn_map, original_size, use_otsu=False):
    """
    Prepare attention heatmap with optional Otsu thresholding

    Args:
        attn_map: attention map tensor [H, W]
        original_size: (height, width) tuple
        use_otsu: whether to apply Otsu thresholding

    Returns:
        attn_resized_np: resized attention map as numpy array
        mask: binary mask (if use_otsu=True) or None
    """
    orig_h, orig_w = original_size
    attn_np = attn_map.squeeze().cpu().numpy()

    # Resize to original size
    attn_resized = Image.fromarray((attn_np * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.Resampling.BILINEAR
    )
    attn_resized_np = np.array(attn_resized)

    if use_otsu:
        otsu_thresh = threshold_otsu(attn_resized_np)
        mask = attn_resized_np > otsu_thresh
        return attn_resized_np, mask

    return attn_resized_np, None


def prepare_attention_heatmap(attn_map, original_size, use_otsu=False):
    """
    Prepare attention heatmap with optional Otsu thresholding

    Args:
        attn_map: attention map tensor [H, W]
        original_size: (height, width) tuple
        use_otsu: whether to apply Otsu thresholding

    Returns:
        attn_resized_np: resized attention map as numpy array [H', W'] in [0,1]
        mask: binary mask (if use_otsu=True) or None
    """
    orig_h, orig_w = original_size
    attn_np = attn_map.squeeze().cpu().numpy()

    # Normalize to [0, 1] before resizing
    if attn_np.max() - attn_np.min() > 1e-6:
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min())
    else:
        attn_np = np.zeros_like(attn_np)

    # Resize to original size
    attn_resized = Image.fromarray((attn_np * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.Resampling.BILINEAR
    )
    attn_resized_np = np.array(attn_resized) / 255.0

    mask = None
    if use_otsu:
        # Apply Otsu thresholding
        otsu_thresh = threshold_otsu(attn_resized_np * 255) / 255.0
        mask = attn_resized_np > otsu_thresh

    return attn_resized_np, mask


def create_heatmap_overlay(img_original_np, attn_resized_np, mask=None, alpha=0.4):
    """
    Create heatmap overlay on image

    Args:
        img_original_np: original image as numpy array [H, W, 3]
        attn_resized_np: resized attention map [H, W]
        mask: optional binary mask
        alpha: blending alpha

    Returns:
        blend_img: blended image [H, W, 3]
    """
    import matplotlib.pyplot as plt

    cam_color = plt.cm.jet(attn_resized_np / 255.0)[..., :3]
    blend_img = img_original_np.copy()

    if mask is not None:
        blend_img[mask] = (1 - alpha) * blend_img[mask] + alpha * cam_color[mask]
    else:
        blend_img = (1 - alpha) * blend_img + alpha * cam_color

    return np.clip(blend_img, 0, 1)


def process_attention_maps(attn_maps, batch_size, original_sizes, use_otsu=False):
    """
    Process batch of attention maps

    Args:
        attn_maps: attention maps [B, H, W]
        batch_size: batch size
        original_sizes: list of (height, width) tuples
        use_otsu: whether to apply Otsu thresholding

    Returns:
        list of (attn_resized_np, mask) tuples
    """
    results = []

    for i in range(batch_size):
        if i < len(original_sizes):
            attn_resized_np, mask = prepare_attention_heatmap(
                attn_maps[i], original_sizes[i], use_otsu
            )
            results.append((attn_resized_np, mask))
        else:
            results.append((None, None))

    return results
