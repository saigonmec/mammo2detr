"""Create a small synthetic "mammography-like" dataset for smoke tests and demos.

    python -m src.tools.make_synthetic_dataset --out /tmp/synth --n 60

Images: dark background with a bright breast-like ellipse; positives contain
1-3 bright blobs ("lesions") with COCO boxes in metadata.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def make_image(rng: np.random.RandomState, h: int, w: int, n_lesions: int) -> tuple[np.ndarray, list[list[float]]]:
    img = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(img, (int(w * 0.35), int(h * 0.5)), (int(w * 0.5), int(h * 0.45)), 0, 0, 360, 0.35, -1)
    img += rng.normal(0, 0.03, size=img.shape).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    boxes = []
    for _ in range(n_lesions):
        r = int(rng.uniform(0.04, 0.12) * min(h, w))
        cx = int(rng.uniform(0.15, 0.7) * w)
        cy = int(rng.uniform(0.15, 0.85) * h)
        cv2.circle(img, (cx, cy), r, 0.95, -1)
        boxes.append([cx - r, cy - r, 2 * r, 2 * r])
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img, boxes


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--size", type=int, nargs=2, default=[256, 192], metavar=("H", "W"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-val", action="store_true", help="also write an explicit 'val' split")
    args = p.parse_args(argv)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    rows = []
    h, w = args.size
    for i in range(args.n):
        positive = i % 2 == 0
        n_les = int(rng.randint(1, 4)) if positive else 0
        img, boxes = make_image(rng, h, w, n_les)
        name = f"img_{i:04d}.png"
        cv2.imwrite(str(out / "images" / name), img)
        frac = i / args.n
        split = "test" if frac >= 0.8 else ("val" if (args.with_val and frac >= 0.65) else "train")
        base = {"image_id": f"img_{i:04d}", "link": f"images/{name}", "split": split, "cancer": int(positive), "patient_id": f"P{i // 2:04d}", "img_width": w, "img_height": h}
        if boxes:
            for b in boxes:
                rows.append({**base, "x": b[0], "y": b[1], "width": b[2], "height": b[3]})
        else:
            rows.append({**base, "x": "", "y": "", "width": "", "height": ""})
    fields = ["image_id", "link", "split", "cancer", "patient_id", "img_width", "img_height", "x", "y", "width", "height"]
    with open(out / "metadata.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {args.n} images and {len(rows)} annotation rows to {out}")
    return out


if __name__ == "__main__":
    main()
