import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from .detr_augment import (
    get_m2_train_augmentation,
    get_m2_test_augmentation,
    get_m2_test_augmentation_no_bbox,
)


def validate_and_clip_bbox(x, y, w, h, img_w, img_h, min_area=100):
    """Validate and clip bbox to image boundaries"""
    x = max(0, float(x))
    y = max(0, float(y))
    w = max(0, float(w))
    h = max(0, float(h))

    x = min(x, img_w - 1)
    y = min(y, img_h - 1)

    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y

    is_valid = w > 0 and h > 0 and (w * h) >= min_area
    return x, y, w, h, is_valid


class M2DETRDataset(Dataset):
    """Dataset for DETR"""

    def __init__(
        self,
        df,
        data_folder,
        positive_transform=None,
        negative_transform=None,
        mode="train",
        img_size=None,
        max_objects=5,
    ):
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.mode = mode
        self.img_size = img_size
        self.max_objects = max_objects
        self.positive_transform = positive_transform
        self.negative_transform = negative_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_folder, row["link"])

        image = cv2.imread(img_path)
        if image is None:
            return {
                "image": torch.zeros(3, 448, 448),
                "label": 0,
                "bboxes": torch.zeros(self.max_objects, 4),
                "bbox_mask": torch.zeros(self.max_objects),
                "num_objects": 0,
                "image_id": str(row["image_id"]),
            }

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        label = int(row["cancer"])

        bbox_list = row.get("bbox_list", [])
        if not isinstance(bbox_list, list):
            bbox_list = []

        bboxes = bbox_list.copy()

        # Apply augmentation
        if len(bboxes) > 0:
            if self.positive_transform:
                bbox_labels = [1] * len(bboxes)
                transformed = self.positive_transform(
                    image=image, bboxes=bboxes, labels=bbox_labels
                )
                image = transformed["image"]
                bboxes = list(transformed["bboxes"])
        else:
            if self.negative_transform:
                transformed = self.negative_transform(image=image)
                image = transformed["image"]
            else:
                if self.img_size:
                    h_t, w_t = (
                        self.img_size
                        if isinstance(self.img_size, tuple)
                        else (self.img_size, self.img_size)
                    )
                else:
                    h_t, w_t = 448, 448
                simple_transform = get_m2_test_augmentation_no_bbox(h_t, w_t)
                transformed = simple_transform(image=image)
                image = transformed["image"]

        # Get target size
        if self.img_size:
            h_img, w_img = self.img_size
        else:
            h_img, w_img = image.shape[1], image.shape[2]

        # Normalize and pad
        normalized_bboxes = []
        for bbox in bboxes[: self.max_objects]:
            x, y, w, h = bbox
            x, y, w, h, is_valid = validate_and_clip_bbox(
                x, y, w, h, w_img, h_img, min_area=1
            )
            if not is_valid:
                continue
            norm_bbox = [
                max(0.0, min(1.0, x / w_img)),
                max(0.0, min(1.0, y / h_img)),
                max(0.0, min(1.0, w / w_img)),
                max(0.0, min(1.0, h / h_img)),
            ]
            if norm_bbox[2] > 0.01 and norm_bbox[3] > 0.01:
                normalized_bboxes.append(norm_bbox)

        num_objects = len(normalized_bboxes)
        padded_bboxes = np.zeros((self.max_objects, 4), dtype=np.float32)
        if num_objects > 0:
            padded_bboxes[:num_objects] = np.array(normalized_bboxes, dtype=np.float32)

        bbox_mask = np.zeros(self.max_objects, dtype=np.float32)
        bbox_mask[:num_objects] = 1.0

        return {
            "image": image,
            "label": label,
            "bboxes": torch.from_numpy(padded_bboxes),
            "bbox_mask": torch.from_numpy(bbox_mask),
            "num_objects": num_objects,
            "image_id": str(row["image_id"]),
        }


def get_image_size_from_config(config_path):
    """Get image size from config file"""
    from src.utils.common import load_config

    config = load_config(config_path)
    img_size = config.get("image_size", 448)
    if isinstance(img_size, str):
        from src.utils.detr_common_utils import parse_img_size

        img_size = parse_img_size(img_size)
    return img_size


def get_weighted_sampler_detr(df, label_col="cancer"):
    """Get weighted sampler for DETR (based on unique images)"""
    from torch.utils.data import WeightedRandomSampler

    labels = df[label_col].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    return sampler


def get_detr_dataloaders(
    train_df,
    test_df,
    data_folder,
    batch_size=16,
    config_path="config/config.yaml",
    img_size=None,
    mode="train",
    max_objects=5,
):
    """Get DETR dataloaders"""

    if img_size is None:
        img_size = get_image_size_from_config(config_path)

    if isinstance(img_size, int):
        height = width = img_size
    else:
        height, width = img_size

    # Get transforms
    positive_train_transform, negative_train_transform = get_m2_train_augmentation(
        height, width
    )
    test_transform = get_m2_test_augmentation(height, width)

    # Create datasets
    train_dataset = M2DETRDataset(
        train_df,
        data_folder,
        positive_transform=positive_train_transform,
        negative_transform=negative_train_transform,
        mode="train",
        img_size=(height, width),
        max_objects=max_objects,
    )

    test_dataset = M2DETRDataset(
        test_df,
        data_folder,
        positive_transform=test_transform,
        negative_transform=None,
        mode="test",
        img_size=(height, width),
        max_objects=max_objects,
    )

    num_workers = 0 if mode == "test" else 4

    sampler = get_weighted_sampler_detr(train_df, label_col="cancer")

    print(f"[INFO] Dataset: {len(train_dataset)} images, Sampler: {len(sampler)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
