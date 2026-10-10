"""Create patient-level, label-stratified train/val/test splits in a metadata CSV.

    python -m src.tools.make_splits --csv /data/metadata.csv --out /data/metadata_split.csv --val 0.1 --test 0.2
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.2)
    p.add_argument("--label-col", default="cancer")
    p.add_argument("--patient-col", default="patient_id")
    p.add_argument("--image-id-col", default="image_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-test", action="store_true", help="keep existing 'test' rows fixed and only split the rest")
    args = p.parse_args(argv)
    df = pd.read_csv(args.csv)
    group_col = args.patient_col if args.patient_col in df.columns else args.image_id_col
    fixed_test = set()
    if args.keep_test and "split" in df.columns:
        fixed_test = set(df.loc[df["split"].astype(str).str.lower() == "test", group_col].astype(str))
    groups: dict[str, int] = defaultdict(int)
    for g, lab in zip(df[group_col].astype(str), pd.to_numeric(df[args.label_col], errors="coerce").fillna(0)):
        groups[g] = max(groups[g], int(lab > 0))
    rng = np.random.RandomState(args.seed)
    assign: dict[str, str] = {g: "test" for g in fixed_test}
    for lab in sorted(set(groups.values())):
        keys = sorted(g for g, l in groups.items() if l == lab and g not in fixed_test)
        rng.shuffle(keys)
        n = len(keys)
        n_test = 0 if args.keep_test else int(round(n * args.test))
        n_val = int(round(n * args.val))
        for i, g in enumerate(keys):
            assign[g] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    df["split"] = df[group_col].astype(str).map(assign)
    df.to_csv(args.out, index=False)
    counts = df.drop_duplicates(args.image_id_col)["split"].value_counts().to_dict()
    print(f"wrote {args.out}: images per split {counts}")


if __name__ == "__main__":
    main()
