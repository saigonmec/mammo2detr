"""Metadata loading: one ``metadata.csv`` with one row per *annotation*.

Required columns: ``image_id``, an image path column (``link`` / ``image_path`` /
``path`` / ``filename``), ``split`` and the image-level label column(s).
Optional: ``x, y, width, height`` (COCO pixel boxes; NaN rows = no box),
``img_width/img_height``, ``patient_id``, a lesion-category column.

Rows are grouped per image into :class:`ImageRecord` objects with pixel
``xyxy`` boxes; labels are factorised deterministically (sorted unique values)
so train/val/test share the same mapping.
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import DataConfig

logger = logging.getLogger(__name__)

IMAGE_PATH_CANDIDATES = ("link", "image_path", "path", "filename", "file_name", "image")


@dataclass
class ImageRecord:
    image_id: str
    path: str
    split: str
    width: int | None
    height: int | None
    patient_id: str | None
    image_labels: dict[str, int]
    boxes_xyxy: np.ndarray  # [n, 4] float32, original-image pixels
    box_labels: np.ndarray  # [n] int64

    @property
    def num_boxes(self) -> int:
        return int(len(self.boxes_xyxy))


@dataclass
class MetadataBundle:
    splits: dict[str, list[ImageRecord]]
    image_class_names: dict[str, list[str]]  # target name -> class names (index order)
    det_class_names: list[str]
    image_class_counts: dict[str, list[int]]  # computed on the train split

    def __getitem__(self, split: str) -> list[ImageRecord]:
        return self.splits.get(split, [])

    def summary(self) -> str:
        lines = ["Dataset summary:"]
        for split, recs in self.splits.items():
            n_pos_box = sum(1 for r in recs if r.num_boxes > 0)
            n_boxes = sum(r.num_boxes for r in recs)
            pats = len({r.patient_id for r in recs if r.patient_id is not None})
            lab = ""
            for name in self.image_class_names:
                c = Counter(r.image_labels[name] for r in recs)
                lab += f" {name}={dict(sorted(c.items()))}"
            lines.append(
                f"  {split:>5}: {len(recs):6d} images | {n_pos_box:5d} with boxes | {n_boxes:5d} boxes | {pats:5d} patients |{lab}"
            )
        lines.append(f"  det classes: {self.det_class_names} | image classes: {self.image_class_names}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _find_col(df: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"None of the columns {candidates} found in metadata (have: {list(df.columns)})")
    return None


def _factorize(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Deterministic label encoding: integer-valued numerics keep their order; strings -> sorted unique."""
    if pd.api.types.is_numeric_dtype(series):
        vals = series.astype(float)
        if np.all(np.mod(vals.values, 1) == 0):
            uniq = sorted(int(v) for v in vals.unique())
            mapping = {v: i for i, v in enumerate(uniq)}
            return vals.astype(int).map(mapping).values, [str(v) for v in uniq]
    uniq = sorted(series.astype(str).unique().tolist())
    mapping = {v: i for i, v in enumerate(uniq)}
    return series.astype(str).map(mapping).values, uniq


def load_metadata(cfg: DataConfig, seed: int = 42) -> MetadataBundle:
    root = Path(cfg.root)
    csv_path = Path(cfg.metadata_csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"metadata not found: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded %s: %d rows, %d unique images", csv_path, len(df), df[cfg.image_id_col].nunique())

    img_col = cfg.image_col or _find_col(df, IMAGE_PATH_CANDIDATES)
    id_col = cfg.image_id_col
    if id_col not in df.columns:
        raise KeyError(f"image_id column '{id_col}' missing")
    if cfg.split_col not in df.columns:
        raise KeyError(f"split column '{cfg.split_col}' missing")
    df[id_col] = df[id_col].astype(str)
    df[cfg.split_col] = df[cfg.split_col].astype(str).str.strip().str.lower()

    # ---- image-level labels (encoded per target)
    image_class_names: dict[str, list[str]] = {}
    for it in cfg.image_labels:
        if it.name not in df.columns:
            raise KeyError(f"image label column '{it.name}' missing from metadata")
        n_before = len(df)
        df = df[df[it.name].notna()].copy()
        if len(df) != n_before:
            logger.warning("Dropped %d rows with NaN in '%s'", n_before - len(df), it.name)
        codes, names = _factorize(df[it.name])
        df[f"__lab_{it.name}"] = codes
        if len(names) > it.num_classes:
            raise ValueError(f"Target '{it.name}' has {len(names)} classes {names} but config says num_classes={it.num_classes}")
        image_class_names[it.name] = names + [f"class{i}" for i in range(len(names), it.num_classes)]

    # ---- lesion labels
    det_class_names = list(cfg.det_class_names)
    if cfg.lesion_label_col is not None:
        if cfg.lesion_label_col not in df.columns:
            raise KeyError(f"lesion label column '{cfg.lesion_label_col}' missing")
        mask_has = df[cfg.lesion_label_col].notna()
        codes, names = _factorize(df.loc[mask_has, cfg.lesion_label_col])
        if set(names) - set(det_class_names):
            if det_class_names == ["lesion"]:
                det_class_names = names
            else:
                raise ValueError(f"lesion classes in data {names} not covered by det_class_names {det_class_names}")
        remap = {i: det_class_names.index(n) for i, n in enumerate(names)}
        df["__det_lab"] = -1
        df.loc[mask_has, "__det_lab"] = pd.Series(codes, index=df.index[mask_has]).map(remap).values
    else:
        df["__det_lab"] = 0

    # ---- boxes
    bx, by, bw, bh = cfg.bbox_cols
    has_box_cols = all(c in df.columns for c in (bx, by, bw, bh))
    if has_box_cols:
        for c in (bx, by, bw, bh):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        logger.warning("No box columns %s in metadata: training detection without boxes", cfg.bbox_cols)
    wcol = _find_col(df, (cfg.img_width_col, "image_width", "width_px"), required=False)
    hcol = _find_col(df, (cfg.img_height_col, "image_height", "height_px"), required=False)
    pcol = cfg.patient_col if cfg.patient_col and cfg.patient_col in df.columns else None

    records_by_split: dict[str, list[ImageRecord]] = defaultdict(list)
    n_dropped_boxes = 0
    for image_id, g in df.groupby(id_col, sort=True):
        first = g.iloc[0]
        boxes, labels = [], []
        if has_box_cols:
            for _, r in g.iterrows():
                x, y, w, h = r[bx], r[by], r[bw], r[bh]
                if any(pd.isna(v) for v in (x, y, w, h)):
                    continue
                if cfg.bbox_format == "xyxy":
                    x0, y0, x1, y1 = float(x), float(y), float(w), float(h)
                else:
                    x0, y0, x1, y1 = float(x), float(y), float(x) + float(w), float(y) + float(h)
                if x1 - x0 <= 0 or y1 - y0 <= 0:
                    continue
                if (x1 - x0) * (y1 - y0) < cfg.min_box_area:
                    n_dropped_boxes += 1
                    continue
                boxes.append([x0, y0, x1, y1])
                labels.append(int(r["__det_lab"]) if r["__det_lab"] >= 0 else 0)
        width = int(first[wcol]) if wcol is not None and not pd.isna(first[wcol]) else None
        height = int(first[hcol]) if hcol is not None and not pd.isna(first[hcol]) else None
        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if width is not None and height is not None and len(boxes_np):
            boxes_np[:, [0, 2]] = boxes_np[:, [0, 2]].clip(0, width)
            boxes_np[:, [1, 3]] = boxes_np[:, [1, 3]].clip(0, height)
            keep = ((boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])) >= cfg.min_box_area
            n_dropped_boxes += int((~keep).sum())
            boxes_np = boxes_np[keep]
            labels = [l for l, k in zip(labels, keep) if k]
        if cfg.max_boxes is not None and len(boxes_np) > cfg.max_boxes:
            boxes_np = boxes_np[: cfg.max_boxes]
            labels = labels[: cfg.max_boxes]
        path = str(first[img_col])
        if not Path(path).is_absolute():
            path = str(root / path)
        rec = ImageRecord(
            image_id=str(image_id),
            path=path,
            split=str(first[cfg.split_col]),
            width=width,
            height=height,
            patient_id=str(first[pcol]) if pcol is not None and not pd.isna(first[pcol]) else None,
            image_labels={it.name: int(first[f"__lab_{it.name}"]) for it in cfg.image_labels},
            boxes_xyxy=boxes_np,
            box_labels=np.asarray(labels, dtype=np.int64),
        )
        records_by_split[rec.split].append(rec)
    if n_dropped_boxes:
        logger.warning("Dropped %d degenerate/small boxes (min_box_area=%.1f)", n_dropped_boxes, cfg.min_box_area)

    splits = dict(records_by_split)
    train = splits.get(cfg.train_split, [])
    if not train:
        raise ValueError(f"No images in split '{cfg.train_split}'. Available splits: {sorted(splits)}")
    if cfg.val_split not in splits and cfg.val_fraction > 0:
        primary = cfg.image_labels[0].name if cfg.image_labels else None
        tr, va = patient_level_split(train, cfg.val_fraction, seed, primary)
        splits[cfg.train_split], splits[cfg.val_split] = tr, va
        for r in va:
            r.split = cfg.val_split
        logger.info("No '%s' split in metadata: carved %d images (%.0f%%) from train at patient level", cfg.val_split, len(va), 100 * cfg.val_fraction)

    counts: dict[str, list[int]] = {}
    for it in cfg.image_labels:
        c = Counter(r.image_labels[it.name] for r in splits[cfg.train_split])
        counts[it.name] = [c.get(i, 0) for i in range(it.num_classes)]
    bundle = MetadataBundle(splits=splits, image_class_names=image_class_names, det_class_names=det_class_names, image_class_counts=counts)
    logger.info("%s", bundle.summary())
    return bundle


def patient_level_split(records: list[ImageRecord], fraction: float, seed: int, stratify_target: str | None) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Deterministic group split (by patient when available) stratified on the patient-level max label."""
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        groups[r.patient_id if r.patient_id is not None else r.image_id].append(r)
    keys = sorted(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)
    by_label: dict[int, list[str]] = defaultdict(list)
    for k in keys:
        lab = max(r.image_labels[stratify_target] for r in groups[k]) if stratify_target else 0
        by_label[lab].append(k)
    val_keys: set[str] = set()
    for lab, ks in by_label.items():
        n_val = int(math.ceil(len(ks) * fraction)) if len(ks) > 1 else 0
        val_keys.update(ks[:n_val])
    train_out = [r for k in keys if k not in val_keys for r in groups[k]]
    val_out = [r for k in keys if k in val_keys for r in groups[k]]
    return train_out, val_out


def records_to_dataframe(records: list[ImageRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in records:
        rows.append({"image_id": r.image_id, "path": r.path, "split": r.split, "num_boxes": r.num_boxes, **r.image_labels})
    return pd.DataFrame(rows)
