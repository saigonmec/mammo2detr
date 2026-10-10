"""Drawing helpers (OpenCV) for ground truth / predictions / attention maps."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _color_for_score(score: float) -> tuple[int, int, int]:
    if score >= 0.8:
        return (255, 0, 0)  # red (RGB)
    if score >= 0.5:
        return (255, 165, 0)  # orange
    return (255, 255, 0)  # yellow


def draw_detections(
    image_rgb: np.ndarray,
    gt_boxes: np.ndarray | None,
    pred_boxes: np.ndarray | None,
    pred_scores: np.ndarray | None,
    pred_labels: np.ndarray | None = None,
    class_names: list[str] | None = None,
    score_threshold: float = 0.3,
    title: str | None = None,
    thickness: int | None = None,
) -> np.ndarray:
    img = np.ascontiguousarray(image_rgb).copy()
    h, w = img.shape[:2]
    t = thickness or max(2, int(round(min(h, w) / 300)))
    fs = max(0.5, min(h, w) / 900)
    if gt_boxes is not None:
        for b in np.asarray(gt_boxes).reshape(-1, 4):
            x0, y0, x1, y1 = [int(round(v)) for v in b]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), t)
    if pred_boxes is not None and pred_scores is not None:
        order = np.argsort(np.asarray(pred_scores))  # draw high scores last (on top)
        for i in order:
            s = float(pred_scores[i])
            if s < score_threshold:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in np.asarray(pred_boxes[i])]
            c = _color_for_score(s)
            cv2.rectangle(img, (x0, y0), (x1, y1), c, t)
            label = f"{s:.2f}"
            if pred_labels is not None and class_names and len(class_names) > 1:
                label = f"{class_names[int(pred_labels[i])]} {s:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
            y_text = max(th + 4, y0 - 4)
            cv2.rectangle(img, (x0, y_text - th - 4), (x0 + tw + 4, y_text + 2), (0, 0, 0), -1)
            cv2.putText(img, label, (x0 + 2, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, c, max(1, t // 2), cv2.LINE_AA)
    if title and w >= 256:  # skip titles on thumbnails where they cannot fit
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
        cv2.rectangle(img, (0, 0), (tw + 10, th + 12), (0, 0, 0), -1)
        cv2.putText(img, title, (5, th + 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, t // 2), cv2.LINE_AA)
    return img


def overlay_heatmap(image_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [h, w] heat map (any range) onto an RGB image."""
    h, w = image_rgb.shape[:2]
    hm = np.asarray(heat, dtype=np.float32)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return np.clip((1 - alpha) * image_rgb + alpha * color, 0, 255).astype(np.uint8)


def save_prediction_image(path: str | Path, image_rgb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
