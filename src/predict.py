"""Run inference on a folder of images (or a single image) and save overlays + CSV.

    python -m src.predict --checkpoint runs/<run>/best.pt --input /folder/of/pngs --output /out --attention
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import Config
from .data.dataset import load_rgb
from .data.transforms import build_eval_transform, final_resize
from .engine.postprocess import PostProcess
from .models import build_model
from .utils.checkpoint import extract_model_state, load_checkpoint
from .utils.logger import setup_logging
from .utils.misc import resolve_device
from .viz.draw import draw_detections, overlay_heatmap, save_prediction_image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def preprocess(image: np.ndarray, cfg: Config, transform) -> tuple[torch.Tensor, torch.Tensor, dict]:
    out = transform(image=image, bboxes=[], labels=[])
    canvas, _, pad_mask, (vh, vw), _ = final_resize(out["image"], np.zeros((0, 4), np.float32), cfg.data)
    mean = np.asarray(cfg.data.mean, np.float32).reshape(1, 1, 3)
    std = np.asarray(cfg.data.std, np.float32).reshape(1, 1, 3)
    img = (canvas.astype(np.float32) / 255.0 - mean) / std
    t = {"size": torch.tensor([vh, vw], dtype=torch.float32), "orig_size": torch.tensor(image.shape[:2], dtype=torch.float32)}
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), torch.from_numpy(pad_mask), t


def attention_map(model, outputs: dict, query_idx: torch.Tensor, image_index: int = 0) -> np.ndarray | None:
    """Cross-attention of selected queries from the last decoder layer as a [h, w] map."""
    layer = model.decoder_layers[-1]
    attn = layer.last_attn
    shapes = outputs["spatial_shapes"]
    if attn is None:
        return None
    if isinstance(attn, tuple):  # deformable: (locations [N,Lq,M,L,P,2], weights [N,Lq,M,L,P])
        locs, w = attn
        h, wd = shapes[-1]
        heat = np.zeros((h, wd), dtype=np.float32)
        for q in query_idx.tolist():
            l_ = locs[image_index, q].reshape(-1, 2).cpu().numpy()
            ww = w[image_index, q].reshape(-1).cpu().numpy()
            xs = np.clip((l_[:, 0] * wd).astype(int), 0, wd - 1)
            ys = np.clip((l_[:, 1] * h).astype(int), 0, h - 1)
            np.add.at(heat, (ys, xs), ww)
        return heat
    a = attn[image_index, query_idx].mean(0)  # [S]
    h, wd = shapes[0]
    return a[: h * wd].reshape(h, wd).float().cpu().numpy()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MammoDETR inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="image file or folder")
    p.add_argument("--output", required=True)
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--attention", action="store_true", help="also save decoder cross-attention overlays")
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args(argv)

    out_dir = Path(args.output)
    setup_logging(out_dir)
    log = logging.getLogger("src.predict")
    ckpt = load_checkpoint(args.checkpoint)
    cfg = Config.from_dict(dict(ckpt["config"]))
    device = resolve_device(args.device or cfg.device)
    model = build_model(cfg)
    state, source = extract_model_state(ckpt, prefer_ema=not args.no_ema)
    model.load_state_dict(state)
    model.to(device).eval()
    model.set_store_attention(args.attention)
    image_class_names = ckpt.get("image_class_names", {})
    det_class_names = ckpt.get("det_class_names", cfg.data.det_class_names)
    transform = build_eval_transform(cfg.data)
    post = PostProcess(args.topk, 0.0, cfg.eval.nms_iou)
    images = list_images(Path(args.input))
    log.info("Loaded %s weights; %d images", source, len(images))
    rows = []
    for path in images:
        image = load_rgb(str(path))
        x, m, t = preprocess(image, cfg, transform)
        with torch.no_grad():
            outputs = model(x[None].to(device), m[None].to(device))
        res = post(outputs, [t])[0]
        boxes, scores, labels = res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res["labels"].cpu().numpy()
        row = {"image": str(path), "max_box_score": float(scores.max()) if len(scores) else 0.0, "n_boxes": int((scores >= args.score_threshold).sum())}
        title = ""
        for name, logits in outputs.get("image_logits", {}).items():
            prob = torch.softmax(logits[0].float(), -1).cpu().numpy()
            names = image_class_names.get(name) or [str(i) for i in range(len(prob))]
            for i, pv in enumerate(prob):
                row[f"prob_{name}_{names[i]}"] = float(pv)
            row[f"pred_{name}"] = names[int(prob.argmax())]
            title += f"{name}: {names[int(prob.argmax())]} ({prob.max():.2f}) "
        for j, (b, s, l) in enumerate(zip(boxes, scores, labels)):
            if s >= args.score_threshold:
                row[f"box{j}"] = f"{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}:{s:.3f}:{det_class_names[int(l)] if int(l) < len(det_class_names) else l}"
        rows.append(row)
        vis = draw_detections(image, None, boxes, scores, labels, det_class_names, args.score_threshold, title.strip())
        save_prediction_image(out_dir / "overlays" / f"{path.stem}.jpg", vis)
        if args.attention:
            keep = res["query_idx"][res["scores"] >= args.score_threshold]
            if len(keep) == 0:
                keep = res["query_idx"][:1]
            heat = attention_map(model, outputs, keep.cpu())
            if heat is not None:
                vh, vw = t["size"].tolist()
                # crop heat map to the valid region before overlaying on the original image
                hh, hw = heat.shape
                H, W = cfg.data.img_size
                heat = heat[: max(1, int(round(hh * vh / H))), : max(1, int(round(hw * vw / W)))]
                save_prediction_image(out_dir / "attention" / f"{path.stem}.jpg", draw_detections(overlay_heatmap(image, heat), None, boxes, scores, labels, det_class_names, args.score_threshold, title.strip()))
    if rows:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "image", k))
        with open(out_dir / "predictions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    log.info("Wrote %d predictions to %s", len(rows), out_dir)


if __name__ == "__main__":
    main()
