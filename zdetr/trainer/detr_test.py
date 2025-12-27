"""
Testing script for DETR model
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from zdetr.data.detr_data import get_detr_dataloaders
from zdetr.data.detr_data_pre import load_detr_metadata
from zdetr.models.detr_model import get_detr_model
from zdetr.utils.common import load_config, get_arg_or_config
from zdetr.utils.detr_common_utils import parse_img_size
from zdetr.utils.detr_gradcam_utils import get_gradcam_layer, gradcam
from zdetr.utils.detr_data_utils import load_image_metadata_with_bboxes
from zdetr.utils.detr_viz_utils import visualize_detr_result
from zdetr.utils.detr_test_utils import (
    evaluate_detr_model,
    compute_classification_metrics,
    print_test_metrics,
)


def save_full_model(
    model,
    model_type,
    output,
    model_filename,
    actual_input_size,
    num_queries,
    gradcam_layer,
    test_metrics,
    device,
):
    """Save full model with metadata"""
    if isinstance(actual_input_size, int):
        actual_input_size = (actual_input_size, actual_input_size)

    dummy_input = torch.randn(1, 3, actual_input_size[0], actual_input_size[1]).to(
        device
    )

    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    model_to_save.eval()
    with torch.no_grad():
        start = time.time()
        _ = model_to_save(dummy_input)
        inference_time = time.time() - start

    # Thêm tên model và số lượng tham số
    model_name = type(model_to_save).__name__
    num_params = sum(p.numel() for p in model_to_save.parameters())

    full_model_dir = os.path.join(output, "models")
    os.makedirs(full_model_dir, exist_ok=True)
    full_model_path = os.path.join(full_model_dir, f"{model_filename}_full.pth")

    model_metadata = {
        "model": model_to_save,
        "model_type": model_type,
        "model_name": model_name,
        "num_params": num_params,
        "input_size": actual_input_size,
        "num_queries": num_queries,
        "gradcam_layer": gradcam_layer,
        "test_metrics": test_metrics,
        "inference_time": inference_time,
    }

    try:
        torch.save(model_metadata, full_model_path)
        print(f"✅ Saved full DETR model to: {full_model_path}")
        print(f"   Model name: {model_metadata['model_name']}")
        print(f"   Model type: {model_metadata['model_type']}")
        print(f"   Num params: {model_metadata['num_params']}")
        print(f"   Input size: {model_metadata['input_size']}")
        print(f"   Num queries: {model_metadata['num_queries']}")
        print(f"   GradCAM layer: {model_metadata['gradcam_layer']}")
        print(f"   Inference time: {model_metadata['inference_time']:.4f}s")
        print(f"   Test Accuracy: {model_metadata['test_metrics']['accuracy']:.2f}%")
        print(f"   Test AUC: {model_metadata['test_metrics'].get('auc', 0.0):.4f}")
        print(f"   Test IoU: {model_metadata['test_metrics']['iou'] * 100:.2f}%")
        print(
            f"   Test mAP@0.5: {model_metadata['test_metrics'].get('map50', 0.0) * 100:.2f}%"
        )
        print(
            f"   Test mAP@0.25: {model_metadata['test_metrics'].get('map25', 0.0) * 100:.2f}%"
        )
        print(
            f"   Test Recall@IoU=0.25: {model_metadata['test_metrics'].get('recall_iou25', 0.0) * 100:.2f}%"
        )
    except Exception as e:
        print(f"⚠️ Error saving full model: {e}")


def generate_visualizations(
    model,
    test_loader,
    test_df,
    image_info,
    class_names,
    output,
    model_filename,
    batch_size,
    actual_input_size,
    obj_threshold,
    use_otsu,
    use_gradcam,
    gradcam_layer,
    device,
):
    """Generate visualizations for test set"""
    vis_dir = os.path.join(output, "test_detr", model_filename)
    os.makedirs(vis_dir, exist_ok=True)

    test_image_ids = test_df["image_id"].unique().tolist()
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Visualizing")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + len(images), len(test_image_ids))
            batch_image_ids = test_image_ids[start_idx:end_idx]

            outputs = model(images, return_attention_maps=True)
            cls_logits = outputs["cls_logits"]
            pred_bboxes = outputs["pred_bboxes"]
            pred_obj = torch.sigmoid(outputs["obj_scores"])
            attn_maps = outputs.get("attn_maps", None)
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

                if (
                    original_size is None
                    or attn_maps is None
                    or len(attn_maps.shape) != 3
                ):
                    continue

                # attn_map = attn_maps[i]
                attn_map = spatial_attn[i]

                # GradCAM generation
                gradcam_map = None
                if use_gradcam and gradcam_layer is not None:
                    try:
                        input_tensor = images[i : i + 1].clone().requires_grad_(True)
                        with torch.set_grad_enabled(True):
                            test_model = (
                                model.module
                                if isinstance(model, nn.DataParallel)
                                else model
                            )
                            # ĐÃ TẮT DEBUG: không in ra các layer nữa
                            result = gradcam(
                                test_model,
                                input_tensor,
                                gradcam_layer,
                                class_idx=pred_class,
                            )
                            if (
                                isinstance(result, np.ndarray)
                                and result.ndim == 2
                                and result.dtype == np.uint8
                            ):
                                gradcam_map = result
                            else:
                                # ĐÃ TẮT DEBUG
                                pass
                    except Exception as e:
                        print(f"⚠️ GradCAM failed for {image_id}: {e}")

                save_path = os.path.join(vis_dir, f"{image_id}.png")
                try:
                    visualize_detr_result(
                        images[i],
                        attn_map,
                        pred_bboxes[i],
                        pred_obj[i].squeeze(-1),
                        gt_bbox_list,
                        pred_class,
                        gt_label,
                        pred_prob,
                        save_path,
                        class_names,
                        original_size,
                        image_path=image_path,
                        obj_threshold=obj_threshold,
                        use_otsu=use_otsu,
                        gradcam_map=gradcam_map,
                    )
                except Exception as e:
                    print(f"⚠️ Error visualizing {image_id}: {e}")

    print(f"✅ Saved DETR visualizations to: {vis_dir}")


def run_detr_test(
    data_folder,
    model_type,
    batch_size,
    output,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
    target_column=None,
    num_queries=3,
    max_objects=3,
    save_visualizations=True,
    only_viz=False,
    sample_viz=False,
    obj_threshold=0.5,
    use_otsu=True,  # Changed: default=True
    use_gradcam=False,
):
    """Run DETR testing"""
    config = load_config(config_path)

    train_df, test_df, class_names = load_detr_metadata(
        data_folder, config_path, target_column
    )
    _, _, image_info = load_image_metadata_with_bboxes(data_folder)

    if class_names and isinstance(class_names[0], int):
        class_names = [str(c) for c in class_names]

    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Test samples: {len(test_df)}")

    _, test_loader = get_detr_dataloaders(
        train_df,
        test_df,
        data_folder,
        batch_size=batch_size,
        config_path=config_path,
        img_size=img_size,
        mode="test",
        max_objects=max_objects,
    )

    model = get_detr_model(
        model_type=model_type, num_classes=len(class_names), num_queries=num_queries
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not pretrained_model_path:
        print("⚠️ No pretrained model path provided!")
        return

    try:
        model.load_state_dict(
            torch.load(pretrained_model_path, map_location=device), strict=False
        )
        print(f"✅ Loaded pretrained model from {pretrained_model_path}")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    try:
        sample_batch = next(iter(test_loader))
        actual_input_size = (
            sample_batch["image"].shape[2],
            sample_batch["image"].shape[3],
        )
    except:
        actual_input_size = img_size if img_size else (448, 448)

    model_filename = os.path.basename(pretrained_model_path).replace(".pth", "")

    # GradCAM setup
    gradcam_layer = None
    if use_gradcam:
        test_model = model.module if isinstance(model, nn.DataParallel) else model
        gradcam_layer = get_gradcam_layer(test_model, model_type.lower())
        if gradcam_layer:
            print(f"✓ GradCAM enabled: {gradcam_layer}")
        else:
            use_gradcam = False

    # Sample viz
    if sample_viz:
        from src.trainer.train_m2 import sample_viz_batches

        viz_dir = os.path.join(output, "test_sample_detr", model_filename)
        sample_viz_batches(test_loader, viz_dir, class_names, num_batches=5)

    # Evaluation
    if not only_viz:
        print("\n" + "=" * 50)
        print("Evaluation on Test Set (DETR)")
        print("=" * 50)

        results = evaluate_detr_model(model, test_loader, device)

        metrics = compute_classification_metrics(
            results["preds"], results["labels"], results["probs"], class_names
        )
        # metrics["accuracy"] = results["accuracy"]

        # Gộp các chỉ số khác vào dict lưu model
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

        save_full_model(
            model,
            model_type,
            output,
            model_filename,
            actual_input_size,
            num_queries,
            gradcam_layer,
            all_metrics,  # truyền dict đã gộp đủ chỉ số
            device,
        )

    # Visualization
    if save_visualizations:
        print("\n" + "=" * 50)
        print("Generate Visualizations (DETR)")
        print("=" * 50)

        generate_visualizations(
            model,
            test_loader,
            test_df,
            image_info,
            class_names,
            output,
            model_filename,
            batch_size,
            actual_input_size,
            obj_threshold,
            use_otsu,
            use_gradcam,
            gradcam_layer,
            device,
        )

    if not only_viz:
        return results["accuracy"] / 100, results["iou"]
    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--img_size", type=str)
    parser.add_argument("--target_column", type=str)
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument("--max_objects", type=int, default=3)
    parser.add_argument("--no_viz", action="store_true")
    parser.add_argument("--only_viz", action="store_true")
    parser.add_argument("--sample_viz", action="store_true")
    parser.add_argument("--obj_threshold", type=float, default=0.5)
    parser.add_argument(
        "--no_otsu", action="store_true"
    )  # Changed: use --no_otsu to disable
    parser.add_argument("--gradcam", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)

    run_detr_test(
        data_folder=get_arg_or_config(
            args.data_folder, config.get("data_folder"), None
        ),
        model_type=get_arg_or_config(args.model_type, config.get("model_type"), None),
        batch_size=get_arg_or_config(args.batch_size, config.get("batch_size"), 16),
        output=get_arg_or_config(args.output, config.get("output"), "output"),
        config_path=args.config,
        img_size=parse_img_size(
            get_arg_or_config(args.img_size, config.get("image_size"), None)
        ),
        pretrained_model_path=args.pretrained_model_path,
        target_column=get_arg_or_config(
            args.target_column, config.get("target_column"), None
        ),
        num_queries=args.num_queries,
        max_objects=args.max_objects,
        save_visualizations=not args.no_viz,
        only_viz=args.only_viz,
        sample_viz=args.sample_viz,
        obj_threshold=args.obj_threshold,
        use_otsu=not args.no_otsu,  # Changed: invert logic
        use_gradcam=args.gradcam,
    )
