"""Dataset + collate producing DETR-style targets."""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..config import DataConfig
from ..utils.box_ops import box_xyxy_to_cxcywh
from .metadata import ImageRecord
from .transforms import final_resize

logger = logging.getLogger(__name__)


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class MammoDetectionDataset(Dataset):
    def __init__(self, records: list[ImageRecord], cfg: DataConfig, transform=None):
        self.records = list(records)
        self.cfg = cfg
        self.transform = transform
        self.mean = np.asarray(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(cfg.std, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.records)

    def _prepare_boxes(self, rec: ImageRecord, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        boxes = rec.boxes_xyxy.astype(np.float32).reshape(-1, 4).copy()
        labels = rec.box_labels.astype(np.int64).copy()
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            keep = ((boxes[:, 2] - boxes[:, 0]) > 1) & ((boxes[:, 3] - boxes[:, 1]) > 1)
            boxes, labels = boxes[keep], labels[keep]
        return boxes, labels

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, dict[str, Any]]:
        rec = self.records[idx]
        image = load_rgb(rec.path)
        h0, w0 = image.shape[:2]
        boxes, labels = self._prepare_boxes(rec, h0, w0)

        if self.transform is not None:
            out = self.transform(image=image, bboxes=boxes.tolist(), labels=labels.tolist())
            image = out["image"]
            boxes = np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.asarray(out["labels"], dtype=np.int64)
        canvas, boxes, pad_mask, (vh, vw), (sx, sy) = final_resize(image, boxes, self.cfg)

        img = (canvas.astype(np.float32) / 255.0 - self.mean) / self.std
        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask_t = torch.from_numpy(pad_mask)

        boxes_t = torch.from_numpy(boxes)
        if len(boxes_t):
            norm = boxes_t / torch.tensor([vw, vh, vw, vh], dtype=torch.float32)
            boxes_t = box_xyxy_to_cxcywh(norm.clamp(0, 1))
            keep = (boxes_t[:, 2] > 1e-4) & (boxes_t[:, 3] > 1e-4)
            boxes_t, labels = boxes_t[keep], labels[keep.numpy()]
        target: dict[str, Any] = {
            "boxes": boxes_t.reshape(-1, 4),
            "labels": torch.from_numpy(labels).long(),
            "image_labels": {k: torch.tensor(v, dtype=torch.long) for k, v in rec.image_labels.items()},
            "image_id": rec.image_id,
            "orig_size": torch.tensor([h0, w0], dtype=torch.float32),
            "size": torch.tensor([vh, vw], dtype=torch.float32),
            "path": rec.path,
        }
        return img_t, mask_t, target


def collate_fn(batch: list[tuple[Tensor, Tensor, dict[str, Any]]]) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    targets = [b[2] for b in batch]
    return images, masks, targets
