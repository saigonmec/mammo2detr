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

    # Debug print for predicted bboxes
    print(f"[DEBUG] Predicted bboxes (normalized): {filtered_bboxes}")
    print(
        f"[DEBUG] Predicted bboxes shape: {filtered_bboxes.shape}, type: {type(filtered_bboxes)}"
    )
    print(f"[DEBUG] Predicted scores: {filtered_scores}")

    # Rescale bboxes to original image size
    pixel_bboxes = [rescale_bbox(bbox, original_size) for bbox in filtered_bboxes]
    print(f"[DEBUG] Predicted bboxes (pixel): {pixel_bboxes}")

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
                    "confidence_score",
                    "num_objects",
                ],
            )
            w.writeheader()
            w.writerows(rows)

    # print(f"✅ Saved outputs to: {out_root}")
    return out_root


def _evaluate_from_dataframe(
    model: nn.Module,
    test_df,
    data_folder_path: Path,
    image_info: Dict,
    class_names: List[str],
    output_root: Path,
    device: torch.device,
    input_size: Tuple[int, int],
    obj_threshold: float = 0.5,
    use_otsu: bool = True,
    use_gradcam: bool = False,
    gradcam_layer: Optional[str] = None,
):
    """
    Evaluate model directly from test_df without using dataloader
    Processes each image individually to save predictions and visualizations
    """
    # print("[DEBUG] Start _evaluate_from_dataframe")
    preprocess = build_preprocess(input_size)

    rows_by_folder: Dict[Path, List[Dict[str, str]]] = {}
    # print(f"[DEBUG] Number of test samples: {len(test_df)} ")
    # print(test_df.columns)

    for idx, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Evaluating", unit="img"
    ):
        # print(f"[DEBUG] Processing idx={idx}")
        image_id = str(row["image_id"])
        gt_label = int(row["cancer"])
        # print(f"[DEBUG] image_id={image_id}, gt_label={gt_label}")

        # Get image info
        if image_id not in image_info:
            print(f"⚠️ [DEBUG] Image {image_id} not found in image_info, skipping...")
            continue

        info = image_info[image_id]
        # print(f"[DEBUG] info keys: {list(info.keys())}")
        img_path = Path(info["image_path"])
        original_size = info["original_size"]
        gt_bbox_list = info.get("bbx_list", None)
        # Debug original_size và gt_bbox_list
        print(f"[DEBUG] original_size for {image_id}: {original_size}")
        print(f"[DEBUG] gt_bbox_list for {image_id}: {gt_bbox_list}")
        if gt_bbox_list is not None:
            print(
                f"[DEBUG] gt_bbox_list type: {type(gt_bbox_list)}, shape: {np.array(gt_bbox_list).shape}"
            )
            print(f"[DEBUG] gt_bbox_list values: {gt_bbox_list}")

        # Determine relative path FIRST (before any continue)
        rel_path = img_path.relative_to(data_folder_path)
        rel_dir = rel_path.parent
        out_dir = output_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if not img_path.exists():
            print(f"⚠️ [DEBUG] Image file not found: {img_path}, skipping...")
            continue

        # Run prediction
        try:
            result = predict_detr_image(
                model,
                img_path,
                preprocess,
                device,
                obj_threshold=obj_threshold,
                use_gradcam=use_gradcam,
                gradcam_layer=gradcam_layer,
            )
        except Exception as e:
            print(f"[DEBUG] Error in predict_detr_image: {e}")
            continue

        img = result["image"]
        pred_class = result["pred_class"]
        confidence = result["confidence"]
        bboxes = result["bboxes"]
        print(f"[DEBUG] Predicted bboxes for image {img_path.name}: {bboxes}")
        scores = result["scores"]
        gradcam_map = result["gradcam_map"]

        # Save original image
        try:
            img.save(out_dir / f"{img_path.stem}.png", format="PNG")
        except Exception as e:
            print(f"[DEBUG] Error saving original image: {e}")

        # Try to draw ground truth bboxes if available
        try:
            if gt_bbox_list is not None and len(gt_bbox_list) > 0:
                # Assume GT format is [x, y, w, h] in pixel (xywh)
                gt_pixel_bboxes = []
                for idx_gt, bbox in enumerate(gt_bbox_list):
                    x, y, w, h = [int(float(v)) for v in bbox]
                    x2 = x + w
                    y2 = y + h
                    gt_pixel_bboxes.append((x, y, x2, y2))
                    print(
                        f"[DEBUG] GT bbox {idx_gt} (xywh): ({x}, {y}, {w}, {h}) -> (x1={x}, y1={y}, x2={x2}, y2={y2})"
                    )
                gt_scores = [1.0] * len(gt_pixel_bboxes)
                img_gt = img.copy()
                img_gt_bbox = draw_predicted_bboxes_on_pil(
                    img_gt,
                    gt_pixel_bboxes,
                    gt_scores,
                    threshold=0.0,
                    width=5,
                )
                img_gt_bbox.save(out_dir / f"{img_path.stem}.png", format="PNG")
        except Exception as e:
            print(f"[DEBUG] Error drawing/saving ground truth bboxes: {e}")

        # Save GradCAM + Otsu + bboxes with confidence
        if gradcam_map is not None:
            try:
                img_gradcam = overlay_gradcam_with_otsu(
                    img, gradcam_map, alpha=0.55, use_otsu=use_otsu
                )
                img_gradcam_bbox = draw_predicted_bboxes_on_pil(
                    img_gradcam, bboxes, scores, obj_threshold, width=5
                )
                img_gradcam_bbox.save(
                    out_dir / f"{img_path.stem}_gradcam.png", format="PNG"
                )

                try:
                    if gt_bbox_list is not None and len(gt_bbox_list) > 0:
                        gt_pixel_bboxes = []
                        for idx_gt, bbox in enumerate(gt_bbox_list):
                            x, y, w, h = [int(float(v)) for v in bbox]
                            x2 = x + w
                            y2 = y + h
                            gt_pixel_bboxes.append((x, y, x2, y2))
                            print(
                                f"[DEBUG] GT bbox {idx_gt} (xywh, gradcam): ({x}, {y}, {w}, {h}) -> (x1={x}, y1={y}, x2={x2}, y2={y2})"
                            )
                        gt_scores = [1.0] * len(gt_pixel_bboxes)
                        img_gradcam_gt = overlay_gradcam_with_otsu(
                            img, gradcam_map, alpha=0.55, use_otsu=use_otsu
                        )
                        img_gradcam_gt_bbox = draw_predicted_bboxes_on_pil(
                            img_gradcam_gt,
                            gt_pixel_bboxes,
                            gt_scores,
                            threshold=0.0,
                            width=5,
                        )
                        img_gradcam_gt_bbox.save(
                            out_dir / f"{img_path.stem}_gradcam.png",
                            format="PNG",
                        )
                except Exception as e:
                    print(f"[DEBUG] Error drawing GT bboxes on GradCAM: {e}")
            except Exception as e:
                print(f"[DEBUG] Error saving gradcam image: {e}")

        # Record results
        class_name = (
            class_names[pred_class]
            if pred_class < len(class_names)
            else str(pred_class)
        )
        gt_class_name = (
            class_names[gt_label] if gt_label < len(class_names) else str(gt_label)
        )

        rows_by_folder.setdefault(rel_dir, []).append(
            {
                "image_id": image_id,
                "ground_truth": str(gt_label),
                "gt_class_name": gt_class_name,
                "label": str(pred_class),
                "pred_class_name": class_name,
                "confidence_score": f"{confidence:.6f}",
                "num_objects": str(len(bboxes)),
                "correct": str(int(pred_class == gt_label)),
            }
        )

    # Write CSV per subfolder
    for rel_dir, rows in rows_by_folder.items():
        csv_path = (
            (output_root / rel_dir / "metadata.csv")
            if str(rel_dir) != "."
            else (output_root / "metadata.csv")
        )
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image_id",
                        "ground_truth",
                        "label",
                        "pred_label",
                        "pred_class_name",
                        "confidence_score",
                        "num_objects",
                        "correct",
                    ],
                )
                w.writeheader()
                w.writerows(rows)
            # print(f"[DEBUG] Saved CSV to {csv_path}")
        except Exception as e:
            print(f"[DEBUG] Error saving CSV: {e}")

    # Gộp tất cả kết quả lại thành một file metadata.csv lớn ở output_root
    all_rows = []
    for rows in rows_by_folder.values():
        all_rows.extend(rows)
    big_csv_path = output_root / "metadata.csv"
    try:
        with open(big_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id",
                    "ground_truth",
                    "label",
                    "pred_label",
                    "pred_class_name",
                    "confidence_score",
                    "num_objects",
                    "correct",
                ],
            )
            w.writeheader()
            w.writerows(all_rows)
        # print(f"✅ Saved merged metadata to: {big_csv_path}")
    except Exception as e:
        print(f"[DEBUG] Error saving merged metadata CSV: {e}")

    print(f"✅ Saved prediction outputs to: {output_root}")


def detr_evaluate_dataset(
    data_folder: str,
    model_path: str,
    output_root: Optional[str] = None,
    device: Optional[str] = None,
    obj_threshold: float = 0.5,
    use_otsu: bool = True,
    use_gradcam: bool = False,
    class_names: Optional[List[str]] = None,
    batch_size: int = 16,
    config_path: str = "config/config.yaml",
    max_objects: int = 3,
    save_visualizations: bool = True,
    only_viz: bool = False,
):
    """
    Evaluate DETR model on dataset with metadata.csv
    Similar to test mode but for inference script
    """
    from zdetr.data.detr_data import get_detr_dataloaders
    from zdetr.data.detr_data_pre import load_detr_metadata
    from zdetr.utils.detr_data_utils import load_image_metadata_with_bboxes
    from zdetr.utils.detr_test_utils import (
        evaluate_detr_model,
        compute_classification_metrics,
        print_test_metrics,
    )

    data_folder_path = Path(data_folder).expanduser().resolve()
    out_root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else data_folder_path.parent / f"{data_folder_path.name}_detr_eval"
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

    # Load metadata - only test split
    try:
        train_df, test_df, loaded_class_names = load_detr_metadata(
            str(data_folder_path), config_path, target_column=None
        )

        _, _, image_info = load_image_metadata_with_bboxes(str(data_folder_path))

        if class_names is None:
            class_names = loaded_class_names

        print(f"✓ Found {len(class_names)} classes: {class_names}")

    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        print("Cannot proceed without valid metadata.csv")
        raise

    # Get dataloader for batch evaluation metrics
    img_size = tuple(input_size) if isinstance(input_size, list) else input_size

    if not only_viz:
        # Get dataloader for batch evaluation metrics
        _, eval_loader = get_detr_dataloaders(
            test_df,
            test_df,
            str(data_folder_path),
            batch_size=batch_size,
            config_path=config_path,
            img_size=img_size,
            mode="test",
            max_objects=max_objects,
        )

        # Evaluate model for metrics
        print("\n" + "=" * 50)
        print("Computing Evaluation Metrics")
        print("=" * 50)

        results = evaluate_detr_model(model, eval_loader, dev)

        metrics = compute_classification_metrics(
            results["preds"], results["labels"], results["probs"], class_names
        )

        all_metrics = dict(metrics)
        all_metrics["iou"] = results.get("iou", 0.0)
        all_metrics["map50"] = results.get("map50", 0.0)
        all_metrics["map25"] = results.get("map25", 0.0)
        all_metrics["recall_iou25"] = results.get("recall_iou25", 0.0)

        print_test_metrics(
            all_metrics,
            all_metrics["iou"],
            all_metrics["map50"],
            all_metrics["map25"],
            all_metrics["recall_iou25"],
        )

        # Save metrics to CSV
        metrics_path = out_root / "evaluation_metrics.csv"
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    writer.writerow([key, f"{value:.4f}"])
                else:
                    writer.writerow([key, str(value)])
        print(f"\n✅ Saved metrics to: {metrics_path}")
    else:
        print("\n⏭️  Skipping metrics computation (--only_viz enabled)")
        all_metrics = None

    # Generate predictions and visualizations from test_df
    print("\n" + "=" * 50)
    print("Generating Predictions and Visualizations")
    print("=" * 50)

    _evaluate_from_dataframe(
        model,
        test_df,
        data_folder_path,
        image_info,
        class_names,
        out_root,
        dev,
        img_size,
        obj_threshold,
        use_otsu,
        use_gradcam and save_visualizations,
        gradcam_layer,
    )

    return out_root, all_metrics


def _generate_eval_visualizations(
    model,
    data_loader,
    df,
    image_info,
    class_names,
    vis_dir,
    batch_size,
    input_size,
    obj_threshold,
    use_otsu,
    use_gradcam,
    gradcam_layer,
    device,
):
    """Generate visualizations for evaluation dataset"""
    from zdetr.utils.detr_viz_utils import visualize_detr_result

    image_ids = df["image_id"].unique().tolist()
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Visualizing")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + len(images), len(image_ids))
            batch_image_ids = image_ids[start_idx:end_idx]

            outputs = model(images, return_attention_maps=True)
            cls_logits = outputs["cls_logits"]
            pred_bboxes = outputs["pred_bboxes"]
            pred_obj = torch.sigmoid(outputs["obj_scores"])
            spatial_attn = outputs.get("spatial_attn", None)

            _, predicted = torch.max(cls_logits, 1)
            probs = torch.softmax(cls_logits, dim=1)

            for i in range(len(images)):
                if i >= len(batch_image_ids):
                    continue

                image_id = str(batch_image_ids[i])
                pred_class = predicted[i].item()
                gt_label = labels[i].item()
                pred_prob = probs[i, pred_class].item()

                if image_id not in image_info:
                    continue

                info = image_info[image_id]
                original_size = info["original_size"]
                image_path = info["image_path"]
                gt_bbox_list = info.get("bbx_list", None)

                # Run prediction
                try:
                    result = predict_detr_image(
                        model,
                        img_path,
                        preprocess,
                        device,
                        obj_threshold=obj_threshold,
                        use_gradcam=use_gradcam,
                        gradcam_layer=gradcam_layer,
                    )
                except Exception as e:
                    print(f"[DEBUG] Error in predict_detr_image: {e}")
                    continue

                img = result["image"]
                pred_class = result["pred_class"]
                confidence = result["confidence"]
                bboxes = result["bboxes"]
                scores = result["scores"]
                gradcam_map = result["gradcam_map"]

                # Save original image
                try:
                    img.save(out_dir / f"{img_path.stem}.png", format="PNG")
                except Exception as e:
                    print(f"[DEBUG] Error saving original image: {e}")

                # Try to draw ground truth bboxes if available
                try:
                    if gt_bbox_list is not None and len(gt_bbox_list) > 0:
                        # Assume GT format is [x, y, w, h] in pixel (xywh)
                        gt_pixel_bboxes = []
                        for idx_gt, bbox in enumerate(gt_bbox_list):
                            x, y, w, h = [int(float(v)) for v in bbox]
                            x2 = x + w
                            y2 = y + h
                            gt_pixel_bboxes.append((x, y, x2, y2))
                            print(
                                f"[DEBUG] GT bbox {idx_gt} (xywh): ({x}, {y}, {w}, {h}) -> (x1={x}, y1={y}, x2={x2}, y2={y2})"
                            )
                        gt_scores = [1.0] * len(gt_pixel_bboxes)
                        img_gt = img.copy()
                        img_gt_bbox = draw_predicted_bboxes_on_pil(
                            img_gt,
                            gt_pixel_bboxes,
                            gt_scores,
                            threshold=0.0,
                            width=5,
                        )
                        img_gt_bbox.save(out_dir / f"{img_path.stem}.png", format="PNG")
                except Exception as e:
                    print(f"[DEBUG] Error drawing/saving ground truth bboxes: {e}")

                # Save GradCAM + Otsu + bboxes with confidence
                if gradcam_map is not None:
                    try:
                        img_gradcam = overlay_gradcam_with_otsu(
                            img, gradcam_map, alpha=0.55, use_otsu=use_otsu
                        )
                        img_gradcam_bbox = draw_predicted_bboxes_on_pil(
                            img_gradcam, bboxes, scores, obj_threshold, width=5
                        )
                        img_gradcam_bbox.save(
                            out_dir / f"{img_path.stem}_gradcam.png", format="PNG"
                        )

                        try:
                            if gt_bbox_list is not None and len(gt_bbox_list) > 0:
                                gt_pixel_bboxes = []
                                for idx_gt, bbox in enumerate(gt_bbox_list):
                                    x, y, w, h = [int(float(v)) for v in bbox]
                                    x2 = x + w
                                    y2 = y + h
                                    gt_pixel_bboxes.append((x, y, x2, y2))
                                    print(
                                        f"[DEBUG] GT bbox {idx_gt} (xywh, gradcam): ({x}, {y}, {w}, {h}) -> (x1={x}, y1={y}, x2={x2}, y2={y2})"
                                    )
                                gt_scores = [1.0] * len(gt_pixel_bboxes)
                                img_gradcam_gt = overlay_gradcam_with_otsu(
                                    img, gradcam_map, alpha=0.55, use_otsu=use_otsu
                                )
                                img_gradcam_gt_bbox = draw_predicted_bboxes_on_pil(
                                    img_gradcam_gt,
                                    gt_pixel_bboxes,
                                    gt_scores,
                                    threshold=0.0,
                                    width=5,
                                )
                                img_gradcam_gt_bbox.save(
                                    out_dir / f"{img_path.stem}_gradcam.png",
                                    format="PNG",
                                )
                        except Exception as e:
                            print(f"[DEBUG] Error drawing GT bboxes on GradCAM: {e}")
                    except Exception as e:
                        print(f"[DEBUG] Error saving gradcam image: {e}")

                # Record results
                class_name = (
                    class_names[pred_class]
                    if pred_class < len(class_names)
                    else str(pred_class)
                )
                gt_class_name = (
                    class_names[gt_label]
                    if gt_label < len(class_names)
                    else str(gt_label)
                )

                rows_by_folder.setdefault(rel_dir, []).append(
                    {
                        "image_id": image_id,
                        "ground_truth": str(gt_label),
                        "gt_class_name": gt_class_name,
                        "label": str(pred_class),
                        "pred_class_name": class_name,
                        "confidence_score": f"{confidence:.6f}",
                        "num_objects": str(len(bboxes)),
                        "correct": str(int(pred_class == gt_label)),
                    }
                )

            # Write CSV per subfolder
            for rel_dir, rows in rows_by_folder.items():
                csv_path = (
                    (output_root / rel_dir / "metadata.csv")
                    if str(rel_dir) != "."
                    else (output_root / "metadata.csv")
                )
                try:
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(
                            f,
                            fieldnames=[
                                "image_id",
                                "ground_truth",
                                "label",
                                "pred_label",
                                "pred_class_name",
                                "confidence_score",
                                "num_objects",
                                "correct",
                            ],
                        )
                        w.writeheader()
                        w.writerows(rows)
                    # print(f"[DEBUG] Saved CSV to {csv_path}")
                except Exception as e:
                    print(f"[DEBUG] Error saving CSV: {e}")

            # Gộp tất cả kết quả lại thành một file metadata.csv lớn ở output_root
            all_rows = []
            for rows in rows_by_folder.values():
                all_rows.extend(rows)
            big_csv_path = output_root / "metadata.csv"
            try:
                with open(big_csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=[
                            "image_id",
                            "ground_truth",
                            "label",
                            "pred_label",
                            "pred_class_name",
                            "confidence_score",
                            "num_objects",
                            "correct",
                        ],
                    )
                    w.writeheader()
                    w.writerows(all_rows)
                # print(f"✅ Saved merged metadata to: {big_csv_path}")
            except Exception as e:
                print(f"[DEBUG] Error saving merged metadata CSV: {e}")

            print(f"✅ Saved prediction outputs to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR Inference on Image Folder")
    parser.add_argument(
        "--input_root", type=str, default=None, help="Input folder with images"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=None,
        help="Dataset folder with metadata.csv (evaluation mode)",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to *_full.pth model"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output folder (default: input_root_detr_predict or data_folder_detr_eval)",
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
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation mode"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Config file path"
    )
    parser.add_argument(
        "--max_objects", type=int, default=3, help="Max objects per image"
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Disable visualizations in evaluation mode",
    )
    parser.add_argument(
        "--only_viz",
        action="store_true",
        default=False,
        help="Only run visualization, skip metrics computation (evaluation mode)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.data_folder is None and args.input_root is None:
        parser.error("Either --data_folder or --input_root must be provided")

    if args.data_folder is not None and args.input_root is not None:
        parser.error("Cannot use both --data_folder and --input_root together")

    # Run in evaluation mode or inference mode
    if args.data_folder:
        print("🔬 Running in EVALUATION mode (with metadata.csv)...")
        detr_evaluate_dataset(
            data_folder=args.data_folder,
            model_path=args.model_path,
            output_root=args.output_root,
            device=args.device,
            obj_threshold=args.obj_threshold,
            use_otsu=not args.no_otsu,
            use_gradcam=not args.no_gradcam,
            class_names=args.class_names,
            batch_size=args.batch_size,
            config_path=args.config,
            max_objects=args.max_objects,
            save_visualizations=not args.no_viz,
            only_viz=args.only_viz,
        )
    else:
        print("🖼️ Running in INFERENCE mode (image folder)...")
        detr_predict_folder(
            input_root=args.input_root,
            model_path=args.model_path,
            output_root=args.output_root,
            device=args.device,
            obj_threshold=args.obj_threshold,
            use_otsu=not args.no_otsu,
            use_gradcam=not args.no_gradcam,
            class_names=args.class_names,
        )
