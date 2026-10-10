"""Augmentation pipelines (albumentations) and the final resize/letterbox step.

One *bbox-aware* pipeline is used for every training image, positives and
negatives alike; the legacy code used two different pipelines, which lets a
classifier pick up augmentation artefacts instead of pathology.
"""
from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ..config import DataConfig  # noqa: E402

BBOX_PARAMS = dict(format="pascal_voc", label_fields=["labels"], clip=True, filter_invalid_bboxes=True)


def _pre_resize(cfg: DataConfig, factor: float = 1.2) -> list[A.BasicTransform]:
    h, w = cfg.img_size
    if cfg.resize_mode == "stretch":
        return [A.Resize(int(h * factor), int(w * factor), interpolation=cv2.INTER_AREA)]
    return [A.LongestMaxSize(max_size=int(max(h, w) * factor), interpolation=cv2.INTER_AREA)]


def build_train_transform(cfg: DataConfig) -> A.Compose:
    a = cfg.aug
    h, w = cfg.img_size
    crop_h, crop_w = int(h * 1.2 * a.bbox_safe_crop_scale), int(w * 1.2 * a.bbox_safe_crop_scale)
    ops: list[A.BasicTransform] = _pre_resize(cfg)
    if a.bbox_safe_crop_p > 0:
        ops.append(A.RandomSizedBBoxSafeCrop(height=crop_h, width=crop_w, erosion_rate=0.0, p=a.bbox_safe_crop_p))
    if a.hflip > 0:
        ops.append(A.HorizontalFlip(p=a.hflip))
    if a.vflip > 0:
        ops.append(A.VerticalFlip(p=a.vflip))
    if a.rotate90 > 0:
        ops.append(A.RandomRotate90(p=a.rotate90))
    if a.affine_p > 0:
        ops.append(
            A.Affine(
                scale=tuple(a.affine_scale),
                translate_percent=(-a.affine_translate, a.affine_translate),
                rotate=(-a.affine_rotate, a.affine_rotate),
                shear=(-a.affine_shear, a.affine_shear),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=a.affine_p,
            )
        )
    if a.brightness_contrast_p > 0:
        ops.append(A.RandomBrightnessContrast(a.brightness_limit, a.contrast_limit, p=a.brightness_contrast_p))
    if a.gamma_p > 0:
        ops.append(A.RandomGamma(gamma_limit=(80, 120), p=a.gamma_p))
    if a.clahe_p > 0:
        ops.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=a.clahe_p))
    if a.blur_sharpen_p > 0:
        ops.append(
            A.OneOf(
                [
                    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.UnsharpMask(blur_limit=(3, 5), alpha=(0.1, 0.3), p=1.0),
                ],
                p=a.blur_sharpen_p,
            )
        )
    if a.noise_p > 0:
        ops.append(A.GaussNoise(std_range=(0.01, 0.05), p=a.noise_p))
    return A.Compose(ops, bbox_params=A.BboxParams(min_visibility=a.min_visibility, **BBOX_PARAMS))


def build_eval_transform(cfg: DataConfig) -> A.Compose:
    return A.Compose(_pre_resize(cfg, factor=1.0), bbox_params=A.BboxParams(**BBOX_PARAMS))


def final_resize(image: np.ndarray, boxes_xyxy: np.ndarray, cfg: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[float, float]]:
    """Resize (stretch) or letterbox ``image`` to ``cfg.img_size``.

    Returns ``(image [H, W, 3], boxes_xyxy (pixels on the canvas), pad_mask [H, W] bool,
    valid_size (h, w), scale (sx, sy))`` where ``scale`` maps input pixels to canvas pixels.
    """
    H, W = cfg.img_size
    h, w = image.shape[:2]
    boxes = boxes_xyxy.astype(np.float32).reshape(-1, 4).copy()
    if cfg.resize_mode == "stretch":
        sx, sy = W / w, H / h
        out = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA if (w > W or h > H) else cv2.INTER_LINEAR)
        mask = np.zeros((H, W), dtype=bool)
        valid = (H, W)
    else:
        s = min(H / h, W / w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        sx, sy = nw / w, nh / h
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        out = np.zeros((H, W, 3), dtype=image.dtype)
        out[:nh, :nw] = resized
        mask = np.ones((H, W), dtype=bool)
        mask[:nh, :nw] = False
        valid = (nh, nw)
    if len(boxes):
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
    return out, boxes, mask, valid, (sx, sy)
