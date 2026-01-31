"""
Preprocessing utilities for DETR detection
Handles grouping multiple bboxes per image_id with robust validation
OPTIMIZED: Vectorized pandas operations, no image loading
"""

import pandas as pd
import numpy as np
import os


def load_detr_metadata(
    data_folder, config_path="config/config.yaml", target_column=None
):
    """
    Load and prepare metadata for DETR training
    This is the main entry point for getting train/test DataFrames

    Args:
        data_folder: Path to data folder containing metadata.csv
        config_path: Path to config file (not used currently)
        target_column: Target column name (default: 'cancer')

    Returns:
        train_df: Training DataFrame with bbox_list
        test_df: Testing DataFrame with bbox_list
        class_names: List of class names
    """
    # Find metadata file
    metadata_path = os.path.join(data_folder, "metadata2.csv")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(data_folder, "metadata.csv")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata file found in {data_folder}")

    # Load and process metadata
    train_df, test_df, _ = prepare_detr_from_metadata(
        metadata_path,
        data_folder=data_folder,
        validate_bbox=True,
        min_area=100,
        verbose=True,
    )

    # Get class names
    target_col = target_column if target_column else "cancer"
    if target_col in train_df.columns:
        class_names = sorted(train_df[target_col].unique().tolist())
    else:
        class_names = [0, 1]  # Default binary classification

    return train_df, test_df, class_names


def validate_bboxes_vectorized(df, min_area=100):
    """
    Validate and clip bboxes using vectorized pandas operations
    NO image loading - uses img_width, img_height from CSV

    Args:
        df: DataFrame with columns [x, y, width, height, img_width, img_height]
        min_area: Minimum bbox area (pixels)

    Returns:
        DataFrame with validated bboxes
    """
    # Ensure float type
    df = df.copy()
    bbox_cols = ["x", "y", "width", "height"]

    # Convert to float and ensure non-negative
    for col in bbox_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df[col] = df[col].clip(lower=0.0)

    # Clip x, y to image boundaries
    df["x"] = df["x"].clip(upper=df["img_width"] - 1)
    df["y"] = df["y"].clip(upper=df["img_height"] - 1)

    # Clip width, height to not exceed image boundaries
    df["width"] = np.minimum(df["width"], df["img_width"] - df["x"])
    df["height"] = np.minimum(df["height"], df["img_height"] - df["y"])

    # Calculate area
    df["bbox_area"] = df["width"] * df["height"]

    # Mark valid bboxes
    df["is_valid_bbox"] = (
        (df["width"] > 0) & (df["height"] > 0) & (df["bbox_area"] >= min_area)
    )

    return df


def get_image_dimensions(df, data_folder):
    """
    Get image dimensions from CSV or by loading first occurrence of each image

    Args:
        df: DataFrame with 'link' column
        data_folder: Root folder

    Returns:
        dict: {image_id: (img_h, img_w)}
    """
    import cv2
    import os

    # Try to get from CSV first
    if "img_width" in df.columns and "img_height" in df.columns:
        # Already have dimensions
        return {}

    # Load dimensions for unique images
    unique_links = df[["image_id", "link"]].drop_duplicates("image_id")
    img_dims = {}

    for _, row in unique_links.iterrows():
        img_path = os.path.join(data_folder, row["link"])
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img_dims[row["image_id"]] = (img.shape[0], img.shape[1])
        except:
            pass

    return img_dims


def remove_large_bboxes(df, area_ratio_thresh=0.9):
    """
    Loại bỏ bbox chiếm trên area_ratio_thresh diện tích ảnh
    Args:
        df: DataFrame có các cột x, y, width, height, img_width, img_height
        area_ratio_thresh: Ngưỡng tỷ lệ diện tích (0.9 = 90%)
    Returns:
        DataFrame đã loại bỏ bbox lớn
    """
    # Tính diện tích bbox và ảnh
    df = df.copy()
    df["bbox_area"] = df["width"] * df["height"]
    df["img_area"] = df["img_width"] * df["img_height"]
    df["bbox_ratio"] = df["bbox_area"] / df["img_area"]
    # Chỉ giữ bbox nhỏ hơn ngưỡng
    df = df[df["bbox_ratio"] <= area_ratio_thresh].copy()
    df = df.drop(columns=["bbox_area", "img_area", "bbox_ratio"])
    return df


def group_bboxes_by_image_vectorized(
    df, data_folder=None, validate_bbox=True, min_area=100
):
    """
    Group multiple bbox annotations by image_id using VECTORIZED operations
    FIXED: Keep ALL images, even those without valid bboxes

    Args:
        df: DataFrame with columns [image_id, link, cancer, x, y, width, height, split, ...]
        data_folder: Root folder (optional, only if img_width/img_height not in CSV)
        validate_bbox: Whether to validate bboxes
        min_area: Minimum bbox area

    Returns:
        DataFrame with one row per image_id, with bbox_list column
    """
    # Step 1: Get image dimensions
    if "img_width" not in df.columns or "img_height" not in df.columns:
        print("  ⚠️ Missing img_width/img_height in CSV, loading from images...")
        if data_folder is None:
            raise ValueError(
                "data_folder required when img_width/img_height not in CSV"
            )

        img_dims = get_image_dimensions(df, data_folder)

        # Add dimensions to dataframe
        df["img_height"] = df["image_id"].map(lambda x: img_dims.get(x, (0, 0))[0])
        df["img_width"] = df["image_id"].map(lambda x: img_dims.get(x, (0, 0))[1])

        # Remove rows with missing dimensions
        df = df[(df["img_height"] > 0) & (df["img_width"] > 0)].copy()

    # Step 2: Validate bboxes using vectorized operations
    if validate_bbox:
        df = validate_bboxes_vectorized(df, min_area=min_area)
        # CHANGED: Don't filter out invalid bboxes yet, keep all rows for aggregation
    else:
        df["is_valid_bbox"] = True

    # Step 3: Group by image_id and aggregate bboxes into lists
    def aggregate_bboxes(group):
        """Aggregate all VALID bboxes for one image into a list"""
        bbox_cols = ["x", "y", "width", "height"]

        # Check if bbox columns exist and have valid values
        if all(col in group.columns for col in bbox_cols):
            # CHANGED: Only take rows with valid bboxes
            valid_rows = (
                group[group["is_valid_bbox"]]
                if "is_valid_bbox" in group.columns
                else group
            )

            if len(valid_rows) > 0:
                bboxes = valid_rows[bbox_cols].values.tolist()
                # Filter out invalid bboxes (all zeros or NaN)
                bboxes = [
                    bbox
                    for bbox in bboxes
                    if not all(v == 0 or pd.isna(v) for v in bbox)
                ]
            else:
                bboxes = []
        else:
            bboxes = []

        return bboxes

    # CHANGED: Group ALL images (not just valid ones)
    grouped = (
        df.groupby("image_id")
        .agg(
            {
                "link": "first",
                "cancer": "first",
                "split": "first",
                "img_width": "first",
                "img_height": "first",
            }
        )
        .reset_index()
    )

    # Add bbox_list column
    grouped["bbox_list"] = df.groupby("image_id").apply(aggregate_bboxes).values
    grouped["num_bboxes"] = grouped["bbox_list"].apply(len)

    # Copy other metadata columns if exist
    metadata_cols = [
        "original_width",
        "original_height",
        "patient_id",
        "laterality",
        "view",
    ]
    for col in metadata_cols:
        if col in df.columns:
            grouped[col] = df.groupby("image_id")[col].first().values

    return grouped


def unique_patients(df, debug_save_path=None):
    """
    Trả về số lượng bệnh nhân duy nhất (đã chuẩn hóa) từ DataFrame.
    Đảm bảo drop_duplicates theo image_id/link trước khi đếm (giống dataloader.py).
    Nếu debug_save_path được truyền vào, sẽ lưu df_unique đã chuẩn hóa ra file csv.
    """

    # Tìm cột patient_id, id, hoặc image_id (không phân biệt hoa thường)
    def find_col(df, candidates):
        df_cols = list(df.columns)
        for col in candidates:
            for c in df_cols:
                if c.lower() == col:
                    return c
        return None

    patient_id_col = find_col(df, ["patient_id", "id", "image_id"])
    if patient_id_col is None:
        return 0

    # Drop duplicates theo link hoặc image_id (giống load_metadata)
    if "link" in df.columns:
        df_unique = df.drop_duplicates(subset=["link"])
    elif "image_id" in df.columns:
        df_unique = df.drop_duplicates(subset=["image_id"])
    else:
        df_unique = df

    # Chuẩn hóa patient_id: loại bỏ tất cả các hậu tố _R, _L, _MLO, _CC xuất hiện ở cuối hoặc giữa chuỗi
    def normalize_pid(val):
        if isinstance(val, str):
            for suffix in ["_R", "_L", "_MLO", "_CC"]:
                val = val.split(suffix)[0]
            return val
        return val

    # Áp dụng chuẩn hóa và lưu ra file nếu cần debug
    norm_pid_series = df_unique[patient_id_col].map(normalize_pid)
    if debug_save_path is not None:
        debug_df = df_unique.copy()
        debug_df["normalized_patient_id"] = norm_pid_series
        debug_df.to_csv(debug_save_path, index=False)
        print(f"⚡️ Saved unique patient debug DataFrame to: {debug_save_path}")

    return norm_pid_series.nunique()


def prepare_detr_dataframe(
    df, data_folder=None, validate_bbox=True, min_area=100, verbose=True
):
    """
    Main preprocessing function for DETR detection
    OPTIMIZED: Uses vectorized pandas operations

    Args:
        df: Raw DataFrame from metadata.csv
        data_folder: Root folder (optional if img_width/img_height in CSV)
        validate_bbox: Whether to validate bboxes
        min_area: Minimum bbox area
        verbose: Print statistics

    Returns:
        Tuple of (train_df, test_df) with grouped bboxes
    """
    if verbose:
        print(f"\n📊 DETR Preprocessing (Vectorized):")
        print(f"  Input: {len(df)} annotations")
    # Loại bỏ bbox quá lớn
    # df = remove_large_bboxes(df, area_ratio_thresh=0.9)
    print(f"  After removing large bboxes: {len(df)} annotations")

    # Group bboxes using vectorized operations
    df_grouped = group_bboxes_by_image_vectorized(
        df, data_folder=data_folder, validate_bbox=validate_bbox, min_area=min_area
    )

    # Split train/test
    train_df = df_grouped[df_grouped["split"] == "train"].copy().reset_index(drop=True)
    test_df = df_grouped[df_grouped["split"] == "test"].copy().reset_index(drop=True)

    if verbose:
        print(f"  Output: {len(df_grouped)} unique images")
        print(f"    Train: {len(train_df)} images")
        print(f"    Test:  {len(test_df)} images")

        # Tính unique patients cho train, test, tổng và lưu ra file để debug
        n_train_patients = unique_patients(
            train_df,  # debug_save_path="train_unique_patients.csv"
        )
        n_test_patients = unique_patients(
            test_df,  # debug_save_path="test_unique_patients.csv"
        )
        combined_df = pd.concat([train_df, test_df])
        n_total_patients = unique_patients(
            combined_df,  # debug_save_path="all_unique_patients.csv"
        )

        pct_train = (
            (n_train_patients / n_total_patients * 100) if n_total_patients > 0 else 0
        )
        pct_test = (
            (n_test_patients / n_total_patients * 100) if n_total_patients > 0 else 0
        )
        print(
            f"    Unique patients: train {n_train_patients} ({pct_train:.1f}%) | test {n_test_patients} ({pct_test:.1f}%) | total {n_total_patients}"
        )

        # Multi-bbox statistics
        train_multi = (train_df["num_bboxes"] > 1).sum()
        test_multi = (test_df["num_bboxes"] > 1).sum()
        print(f"\n  Images with >1 bbox:")
        print(
            f"    Train: {train_multi}/{len(train_df)} ({train_multi / len(train_df) * 100:.1f}%)"
        )
        print(
            f"    Test:  {test_multi}/{len(test_df)} ({test_multi / len(test_df) * 100:.1f}%)"
        )

        if train_multi > 0:
            print(f"  Max bboxes per image (train): {train_df['num_bboxes'].max()}")
        if test_multi > 0:
            print(f"  Max bboxes per image (test):  {test_df['num_bboxes'].max()}")

        # Label distribution with total and percent
        print(f"\n  Label distribution:")
        train_counts = train_df["cancer"].value_counts().sort_index()
        test_counts = test_df["cancer"].value_counts().sort_index()
        all_labels = sorted(set(train_counts.index).union(set(test_counts.index)))
        train_total = len(train_df)
        test_total = len(test_df)
        print("  Label | Train  | %Train  | Test   | %Test   | Total  | %Total")
        for label in all_labels:
            n_train = train_counts.get(label, 0)
            n_test = test_counts.get(label, 0)
            n_total = n_train + n_test
            total = train_total + test_total
            pct_train = (n_train / train_total * 100) if train_total > 0 else 0
            pct_test = (n_test / test_total * 100) if test_total > 0 else 0
            pct_total = (n_total / total * 100) if total > 0 else 0
            print(
                f"    {label}   | {n_train:5d}  | {pct_train:6.2f}% | {n_test:5d}  | {pct_test:6.2f}% | {n_total:5d}  | {pct_total:6.2f}%"
            )

        print("=" * 60)

    return train_df, test_df


def prepare_detr_from_metadata(
    metadata_path, data_folder=None, validate_bbox=True, min_area=100, verbose=True
):
    """
    Load metadata.csv and prepare DETR-ready DataFrames
    OPTIMIZED: No image loading if img_width/img_height in CSV

    Args:
        metadata_path: Path to metadata.csv
        data_folder: Root folder (optional if dimensions in CSV)
        validate_bbox: Whether to validate bboxes
        min_area: Minimum bbox area
        verbose: Print statistics

    Returns:
        Tuple of (train_df, test_df, df_raw)
    """
    import os

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    # Load raw metadata
    df_raw = pd.read_csv(metadata_path)

    # THEN ensure required columns exist (copy if needed and available)
    if "image_width" in df_raw.columns:
        if "img_width" not in df_raw.columns:
            df_raw["img_width"] = df_raw["image_width"]

    if "image_height" in df_raw.columns:
        if "img_height" not in df_raw.columns:
            df_raw["img_height"] = df_raw["image_height"]

    if "image_path" in df_raw.columns:
        if "link" not in df_raw.columns:
            df_raw["link"] = df_raw["image_path"]
    elif "link" in df_raw.columns:
        if "image_path" not in df_raw.columns:
            df_raw["image_path"] = df_raw["link"]

    if verbose:
        print(f"📂 Loading: {metadata_path}")
        print(f"  Raw rows: {len(df_raw)}")
        print(f"  Unique images: {df_raw['image_id'].nunique()}")

        # Check if dimensions are in CSV
        has_dims = "img_width" in df_raw.columns and "img_height" in df_raw.columns
        if has_dims:
            print(f"  ✅ Image dimensions found in CSV (fast mode)")
        else:
            print(f"  ⚠️ Image dimensions NOT in CSV, will load images (slow)")

    # Prepare DETR dataframes using vectorized operations
    train_df, test_df = prepare_detr_dataframe(
        df_raw,
        data_folder=data_folder,
        validate_bbox=validate_bbox,
        min_area=min_area,
        verbose=verbose,
    )

    return train_df, test_df, df_raw


# ============= Example Usage =============
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.data.m2_data_preprocess <metadata.csv> [data_folder]"
        )
        sys.exit(1)

    metadata_path = sys.argv[1]
    data_folder = sys.argv[2] if len(sys.argv) > 2 else None

    # Process metadata
    train_df, test_df, df_raw = prepare_detr_from_metadata(
        metadata_path,
        data_folder=data_folder,
        validate_bbox=True,
        min_area=100,
        verbose=True,
    )

    # Show sample
    print("\n📋 Sample from train_df:")
    print(train_df.head(3))

    # Show multi-bbox examples
    multi_bbox_examples = train_df[train_df["num_bboxes"] > 1].head(3)
    if len(multi_bbox_examples) > 0:
        print("\n📋 Examples with >1 bbox:")
        for _, row in multi_bbox_examples.iterrows():
            print(f"  image_id: {row['image_id']}")
            print(f"    bboxes: {row['bbox_list']}")
