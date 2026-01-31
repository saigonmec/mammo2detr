"""
Data loading utilities for DETR models
- Metadata loading with bounding boxes
"""

import os
import pandas as pd
from PIL import Image


def load_image_metadata_with_bboxes(data_folder):
    """
    Load metadata with bounding box information grouped by image_id

    Returns:
        train_df: DataFrame for training
        test_df: DataFrame for testing
        image_info: dict mapping image_id to {original_size, image_path, bbx_list}
    """
    metadata_path = os.path.join(data_folder, "metadata2.csv")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(data_folder, "metadata.csv")

    if not os.path.exists(metadata_path):
        return None, None, {}

    df = pd.read_csv(metadata_path)

    # THEN ensure required columns exist (copy if needed and available)
    if "image_width" in df.columns:
        if "img_width" not in df.columns:
            df["img_width"] = df["image_width"]

    if "image_height" in df.columns:
        if "img_height" not in df.columns:
            df["img_height"] = df["image_height"]

    if "image_path" in df.columns:
        if "link" not in df.columns:
            df["link"] = df["image_path"]
    elif "link" in df.columns:
        if "image_path" not in df.columns:
            df["image_path"] = df["link"]

    # Create bbx column from x, y, width, height
    if all(col in df.columns for col in ["x", "y", "width", "height"]):
        df["bbx"] = df[["x", "y", "width", "height"]].apply(
            lambda row: [row["x"], row["y"], row["width"], row["height"]], axis=1
        )
        # Group all bboxes by image_id
        bbx_grouped = (
            df.groupby("image_id")["bbx"]
            .apply(list)
            .reset_index()
            .rename(columns={"bbx": "bbx_list"})
        )
        df = df.merge(bbx_grouped, on="image_id", how="left")

    # Create image_info mapping
    image_info = {}
    for image_id in df["image_id"].unique():
        img_rows = df[df["image_id"] == image_id]
        first_row = img_rows.iloc[0]

        img_path = os.path.join(data_folder, first_row["link"])

        # Get original size
        if "original_height" in first_row and "original_width" in first_row:
            orig_h = int(first_row["original_height"])
            orig_w = int(first_row["original_width"])
            original_size = (orig_h, orig_w)
        else:
            if os.path.exists(img_path):
                with Image.open(img_path) as img_orig:
                    original_size = (img_orig.height, img_orig.width)
            else:
                original_size = None

        bbx_list = first_row.get("bbx_list", None)

        image_info[image_id] = {
            "original_size": original_size,
            "image_path": img_path,
            "bbx_list": bbx_list,
        }

    # Split train/test
    train_df = df[df["split"] == "train"] if "split" in df.columns else df
    test_df = df[df["split"] == "test"] if "split" in df.columns else df

    return train_df, test_df, image_info
