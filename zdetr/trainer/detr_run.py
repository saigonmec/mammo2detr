import os
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt

from zdetr.utils.detr_gradcam_utils import gradcam
from zdetr.utils.detr_viz_utils import (
    draw_predicted_bboxes_on_pil,
    overlay_gradcam_with_otsu,
)
from skimage.filters import threshold_otsu


def load_full_detr_model(model_path: str):
    """Load full DETR model with all metadata"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model = checkpoint["model"]
    model.eval()

    input_size = checkpoint.get("input_size", (448, 448))
    model_type = checkpoint.get("model_type", "detr_resnet50")
    model_name = checkpoint.get("model_name", "DETR")
    num_queries = checkpoint.get("num_queries", 3)
    gradcam_layer = checkpoint.get("gradcam_layer", None)

    return model, input_size, model_type, model_name, num_queries, gradcam_layer


def list_images(root: Path, exts=(".png", ".jpg", ".jpeg", ".bmp")) -> List[Path]:
    """Recursively list all images in folder"""
    images = []
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                images.append(Path(dp) / f)
    return images


def build_preprocess(input_size: Tuple[int, int]):
    """Build preprocessing pipeline"""
    return transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize image tensor to numpy array"""
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def rescale_bbox(
    bbox: np.ndarray, img_size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Rescale normalized bbox [cx, cy, w, h] to pixel coordinates [x1, y1, x2, y2]"""
    cx, cy, w, h = bbox
    img_w, img_h = img_size

    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)

    return x1, y1, x2, y2


def predict_detr_image(
    model: nn.Module,
    img_path: Path,
    preprocess,
    device: torch.device,
    obj_threshold: float = 0.5,
    use_gradcam: bool = False,
    gradcam_layer: Optional[str] = None,
) -> Dict:
    """Run DETR prediction on single image"""
    img = Image.open(img_path).convert("RGB")
    original_size = img.size

    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x, return_attention_maps=True)

    cls_logits = outputs["cls_logits"]
    pred_bboxes = outputs["pred_bboxes"]
    obj_scores = torch.sigmoid(outputs["obj_scores"])
    spatial_attn = outputs.get("spatial_attn", None)

    # Get predicted class
    _, predicted = torch.max(cls_logits[0], 0)
    pred_class = predicted.item()
    probs = torch.softmax(cls_logits[0], dim=0)
    confidence = probs[pred_class].item()

    # Filter bboxes by objectness score
    valid_indices = (obj_scores[0].squeeze(-1) >= obj_threshold).cpu().numpy()
    filtered_bboxes = pred_bboxes[0][valid_indices].cpu().numpy()
    filtered_scores = obj_scores[0][valid_indices].cpu().numpy().flatten()

    # Rescale bboxes to original image size
    pixel_bboxes = [rescale_bbox(bbox, original_size) for bbox in filtered_bboxes]

    # Generate GradCAM if requested
    gradcam_map = None
    if use_gradcam and gradcam_layer:
        try:
            input_tensor = x.clone().requires_grad_(True)
            with torch.set_grad_enabled(True):
                result = gradcam(
                    model, input_tensor, gradcam_layer, class_idx=pred_class
                )
                if isinstance(result, np.ndarray) and result.ndim == 2:
                    gradcam_map = result
        except Exception as e:
            print(f"⚠️ GradCAM failed: {e}")

    # Use attention map as fallback
    if gradcam_map is None and spatial_attn is not None:
        attn_map = spatial_attn[0].cpu().numpy()
        if attn_map.ndim == 3:
            attn_map = attn_map.mean(axis=0)
        attn_map = (attn_map - attn_map.min()) / (
            attn_map.max() - attn_map.min() + 1e-8
        )
        gradcam_map = (attn_map * 255).astype(np.uint8)

    return {
        "image": img,
        "pred_class": pred_class,
        "confidence": confidence,
        "bboxes": pixel_bboxes,
        "scores": filtered_scores,
        "gradcam_map": gradcam_map,
        "tensor": x[0],
    }


def detr_predict_folder(
    input_root: str,
    model_path: str,
    output_root: Optional[str] = None,
    device: Optional[str] = None,
    obj_threshold: float = 0.5,
    use_otsu: bool = True,
    use_gradcam: bool = False,
    class_names: Optional[List[str]] = None,
):
    """
    Run DETR inference on all images in folder
    - Saves original image (no bbox)
    - Saves GradCAM with Otsu + colored bbox + confidence scores
    - Saves CSV with predictions per subfolder
    """
    in_root = Path(input_root).expanduser().resolve()
    out_root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else in_root.parent / f"{in_root.name}_detr_predict"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    dev = (
        torch.device(device)
        if device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load model
    model, input_size, model_type, model_name, num_queries, gradcam_layer = (
        load_full_detr_model(model_path)
    )
    model = model.to(dev).eval()

    print(f"✓ Loaded model: {model_name} ({model_type})")
    print(f"✓ Input size: {input_size}")
    print(f"✓ Num queries: {num_queries}")
    print(f"✓ GradCAM layer: {gradcam_layer}")

    preprocess = build_preprocess(tuple(input_size))
    images = list_images(in_root)

    if not class_names:
        class_names = ["negative", "positive"]

    rows_by_folder: Dict[Path, List[Dict[str, str]]] = {}

    for img_path in tqdm(images, desc="DETR Predict", unit="img"):
        rel = img_path.relative_to(in_root)
        rel_dir = rel.parent
        out_dir = out_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run prediction
        result = predict_detr_image(
            model,
            img_path,
            preprocess,
            dev,
            obj_threshold=obj_threshold,
            use_gradcam=use_gradcam,
            gradcam_layer=gradcam_layer,
        )

        img = result["image"]
        pred_class = result["pred_class"]
        confidence = result["confidence"]
        bboxes = result["bboxes"]
        scores = result["scores"]
        gradcam_map = result["gradcam_map"]

        # Save original image (no bboxes)
        img.save(out_dir / f"{img_path.stem}.png", format="PNG")

        # Save GradCAM + Otsu + bboxes with confidence
        if gradcam_map is not None:
            # Apply Otsu thresholding using new function
            img_gradcam = overlay_gradcam_with_otsu(
                img, gradcam_map, alpha=0.55, use_otsu=use_otsu
            )
            # Draw color-coded bboxes with confidence scores using new function
            img_gradcam_bbox = draw_predicted_bboxes_on_pil(
                img_gradcam, bboxes, scores, obj_threshold, width=5
            )
            img_gradcam_bbox.save(
                out_dir / f"{img_path.stem}_gradcam.png", format="PNG"
            )

        # Record results
        class_name = (
            class_names[pred_class]
            if pred_class < len(class_names)
            else str(pred_class)
        )
        rows_by_folder.setdefault(rel_dir, []).append(
            {
                "image_id": img_path.name,
                "label": str(pred_class),
                "class_name": class_name,
                "confidence": f"{confidence:.6f}",
                "num_objects": str(len(bboxes)),
            }
        )

    # Write CSV per subfolder
    for rel_dir, rows in rows_by_folder.items():
        csv_path = (
            (out_root / rel_dir / "result.csv")
            if str(rel_dir) != "."
            else (out_root / "result.csv")
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id",
                    "label",
                    "class_name",
                    "confidence",
                    "num_objects",
                ],
            )
            w.writeheader()
            w.writerows(rows)

    print(f"✅ Saved outputs to: {out_root}")
    return out_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR Inference on Image Folder")
    parser.add_argument(
        "--input_root", type=str, required=True, help="Input folder with images"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to *_full.pth model"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output folder (default: input_root_detr_predict)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--obj_threshold", type=float, default=0.5, help="Objectness threshold"
    )
    parser.add_argument(
        "--no_otsu", action="store_true", help="Disable Otsu thresholding"
    )
    parser.add_argument(
        "--no_gradcam", action="store_true", help="Disable GradCAM overlay"
    )
    parser.add_argument(
        "--class_names", type=str, nargs="+", default=None, help="Class names"
    )

    args = parser.parse_args()

    detr_predict_folder(
        input_root=args.input_root,
        model_path=args.model_path,
        output_root=args.output_root,
        device=args.device,
        obj_threshold=args.obj_threshold,
        use_otsu=not args.no_otsu,
        use_gradcam=not args.no_gradcam,  # GradCAM enabled by default
        class_names=args.class_names,
    )
