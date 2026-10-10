"""Sanity-check a dataset folder before training.

    python -m src.tools.validate_data --data-root /data/mammo [--config cfg.yaml] [--check-images all|sample|none]
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Config
from ..data.metadata import load_metadata
from ..utils.logger import setup_logging


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--check-images", choices=["all", "sample", "none"], default="sample")
    p.add_argument("--report", default=None, help="write a JSON report here")
    args = p.parse_args(argv)
    setup_logging(None)
    log = logging.getLogger("src.validate_data")
    cfg = Config.load(args.config, list(args.overrides) + [f"data.root={args.data_root}"])
    bundle = load_metadata(cfg.data, cfg.seed)
    print(bundle.summary())
    report: dict = {"splits": {}}
    all_recs = [r for recs in bundle.splits.values() for r in recs]
    # patient leakage check
    by_split_patients = {s: {r.patient_id for r in recs if r.patient_id} for s, recs in bundle.splits.items()}
    names = list(by_split_patients)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            common = by_split_patients[names[i]] & by_split_patients[names[j]]
            if common:
                log.warning("PATIENT LEAKAGE: %d patients appear in both '%s' and '%s' (e.g. %s)", len(common), names[i], names[j], sorted(common)[:5])
            report[f"leak_{names[i]}_{names[j]}"] = len(common)
    # box statistics
    areas, ratios = [], []
    for r in all_recs:
        for b in r.boxes_xyxy:
            w, h = b[2] - b[0], b[3] - b[1]
            areas.append(w * h)
            if r.width and r.height:
                ratios.append(np.sqrt(w * h) / np.sqrt(r.width * r.height))
    if areas:
        q = np.percentile(areas, [5, 25, 50, 75, 95])
        log.info("Box area px^2 percentiles 5/25/50/75/95: %s", [int(v) for v in q])
        report["box_area_percentiles"] = [float(v) for v in q]
    if ratios:
        q = np.percentile(ratios, [5, 50, 95])
        log.info("Box size relative to image side (sqrt-area ratio) 5/50/95: %s", [round(float(v), 4) for v in q])
        report["box_rel_size_percentiles"] = [float(v) for v in q]
        small = float(np.mean(np.asarray(ratios) < 0.05))
        log.info("Fraction of boxes smaller than 5%% of the image side: %.1f%% -> consider stride-8/16 levels or larger img_size", 100 * small)
    report["boxes_per_image"] = dict(Counter(r.num_boxes for r in all_recs))
    # image checks
    if args.check_images != "none":
        rng = np.random.RandomState(0)
        recs = all_recs if args.check_images == "all" else [all_recs[i] for i in rng.choice(len(all_recs), min(50, len(all_recs)), replace=False)]
        missing, bad_size, sizes = 0, 0, []
        for r in recs:
            pth = Path(r.path)
            if not pth.exists():
                missing += 1
                continue
            with Image.open(pth) as im:
                w, h = im.size
            sizes.append((h, w))
            if r.width and r.height and (w != r.width or h != r.height):
                bad_size += 1
        log.info("Checked %d images: %d missing, %d with size mismatch vs CSV; image sizes seen: %s", len(recs), missing, bad_size, sorted(set(sizes))[:5])
        report.update({"checked": len(recs), "missing": missing, "size_mismatch": bad_size})
        if missing:
            log.error("Missing image files - fix paths (data.image_col / data.root) before training")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
    return report


if __name__ == "__main__":
    main()
