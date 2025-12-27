"""
Training engine for DETR model
- Hungarian matching
- DETR loss computation
- Training loop with metrics
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import csv
from datetime import datetime
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)

from zdetr.utils.loss import FocalLoss, LDAMLoss, FocalLoss2
from zdetr.utils.plot import plot_metrics, plot_metrics_det
from zdetr.utils.bbox_loss import GIoULoss
from zdetr.utils.detr_common_utils import box_iou
from zdetr.utils.detr_test_utils import (
    detr_compute_iou,
    compute_classification_metrics,
    print_test_metrics,
)
from zdetr.utils.detr_train_utils import HungarianMatcher, DETRCriterion
from zdetr.utils.detr_metrics import compute_detr_metrics, box_iou_batch


def train_detr_model(
    model,
    train_loader,
    test_loader,
    num_epochs=100,
    lr=1e-4,
    device="cpu",
    model_name="detr",
    output="output",
    dataset_folder="None",
    train_df=None,
    patience=50,
    loss_type="ce",
    lambda_bbox=5.0,
    lambda_giou=2.0,
    lambda_obj=1.0,
):
    """Train DETR model"""
    model = model.to(device)

    # Classification loss
    if train_df is not None:
        class_counts = train_df["cancer"].value_counts()
        weights = torch.tensor(
            [
                len(train_df) / (len(class_counts) * class_counts[i])
                for i in range(len(class_counts))
            ],
            dtype=torch.float,
        ).to(device)
    else:
        weights = None

    if loss_type == "focal":
        cls_criterion = FocalLoss(alpha=weights)
    elif loss_type == "ldam":
        cls_criterion = LDAMLoss(
            cls_num_list=train_df["cancer"].value_counts().sort_index().tolist(),
            weight=weights,
        ).to(device)
    elif loss_type == "focal2":
        cls_criterion = FocalLoss2(alpha=weights)
    else:
        cls_criterion = nn.CrossEntropyLoss(weight=weights)

    criterion = DETRCriterion(cls_criterion, lambda_bbox, lambda_giou, lambda_obj, 0.5)

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": lr * 0.1,
        },
    ]
    optimizer = optim.AdamW(param_dicts, weight_decay=1e-4)
    scaler = GradScaler() if device != "cpu" else None

    warmup_epochs = 5
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda e: (e + 1) / warmup_epochs
        if e < warmup_epochs
        else 0.95 ** (e - warmup_epochs),
    )
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )

    # Setup directories
    model_dir = os.path.join(output, "models")
    plot_dir = os.path.join(output, "figures")
    log_dir = os.path.join(output, "logs")
    for d in [model_dir, plot_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    dataset = dataset_folder.split("/")[-1]
    try:
        sample = next(iter(train_loader))
        img_size = (sample["image"].shape[2], sample["image"].shape[3])
    except:
        img_size = (448, 448)

    model_key = f"{dataset}_{img_size[0]}x{img_size[1]}_detr_{model_name}"
    log_file = os.path.join(log_dir, f"{model_key}.csv")

    # Check for existing weights
    print(f"Checking for existing weights in {model_dir} with model_key: {model_key}")
    existing_weights = []
    for fname in os.listdir(model_dir):
        if (
            fname.startswith(model_key)
            and fname.endswith(".pth")
            and not fname.endswith("_full.pth")
        ):
            try:
                parts = fname.replace(".pth", "").split("_")
                acc_part = parts[-2] if len(parts) > 2 else parts[-1]
                recall_part = parts[-1]
                acc_val = float(acc_part) / 10000
                recall_val = float(recall_part) / 10000
                existing_weights.append((fname, acc_val, recall_val))
            except Exception:
                print(f"Skipping invalid model file: {fname}")
    if existing_weights:
        print("Found existing weights:")
        for fname, acc_val, recall_val in existing_weights:
            print(f"  - {fname} (acc: {acc_val:.4f}, recall@0.25: {recall_val:.4f})")
    else:
        print("No existing weights found.")

    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "datetime",
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "train_iou",
                    "test_loss",
                    "test_acc",
                    "test_iou",
                    "test_map50",
                    "test_map25",
                    "recall_iou25",
                    "lr",
                    "patience",
                    "best_acc",
                ]
            )

    train_losses, train_accs, test_losses, test_accs, train_ious, test_ious = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    test_map25s = []
    train_recalls_025 = []
    test_recalls_025 = []
    best_acc, patience_counter, last_lr = 0.0, 0, lr

    # Track best epoch by recall_iou25
    best_recall_iou25 = 0.0
    best_epoch_info = {}

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        epoch_iou, num_bbox_samples = 0.0, 0

        # Thêm các list để gom bbox và obj_scores cho train
        train_pred_boxes, train_gt_boxes, train_obj_scores = [], [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bboxes"].to(device)
            bbox_mask = batch["bbox_mask"].to(device)

            optimizer.zero_grad()

            if scaler:
                with autocast():
                    outputs = model(images)
                    loss_dict = criterion(
                        outputs,
                        {"label": labels, "bboxes": bboxes, "bbox_mask": bbox_mask},
                    )
                    loss = loss_dict["total_loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss_dict = criterion(
                    outputs, {"label": labels, "bboxes": bboxes, "bbox_mask": bbox_mask}
                )
                loss = loss_dict["total_loss"]
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs["cls_logits"], 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if bbox_mask.sum() > 0:
                with torch.no_grad():
                    for i in range(images.size(0)):
                        if bbox_mask[i].sum() > 0:
                            pred = outputs["pred_bboxes"][i, 0:1]
                            tgt = bboxes[i][bbox_mask[i] > 0.5][:1]
                            epoch_iou += detr_compute_iou(pred[0], tgt[0]).item()
                            num_bbox_samples += 1
                            # Gom bbox cho recall@0.25 train
                            train_pred_boxes.append(outputs["pred_bboxes"][i].cpu())
                            train_gt_boxes.append(bboxes[i].cpu())
                            train_obj_scores.append(
                                torch.sigmoid(outputs["obj_scores"][i]).cpu()
                            )

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        avg_train_iou = epoch_iou / max(num_bbox_samples, 1)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        train_ious.append(avg_train_iou)

        # Print training summary (optional, can keep or remove)
        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} "
            f"Train Acc: {epoch_acc:.4f} IoU: {avg_train_iou:.4f} Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # ===== VALIDATION =====
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        all_pred_boxes, all_gt_boxes, all_obj_scores = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bboxes"].to(device)
                bbox_mask = batch["bbox_mask"].to(device)

                outputs = model(images)
                loss_dict = criterion(
                    outputs, {"label": labels, "bboxes": bboxes, "bbox_mask": bbox_mask}
                )

                val_loss += loss_dict["total_loss"].item() * images.size(0)
                probs = torch.softmax(outputs["cls_logits"], dim=1)
                _, predicted = torch.max(outputs["cls_logits"], 1)

                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Collect boxes and scores for mAP computation
                pred_obj = torch.sigmoid(outputs["obj_scores"])
                for i in range(images.size(0)):
                    if bbox_mask[i].sum() > 0:
                        all_pred_boxes.append(outputs["pred_bboxes"][i].cpu())
                        all_gt_boxes.append(bboxes[i].cpu())
                        all_obj_scores.append(pred_obj[i].cpu())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        test_losses.append(val_loss)
        test_accs.append(val_acc)

        # Compute standard DETR metrics using the new metrics module
        bbox_metrics = {}
        if len(all_pred_boxes) > 0:
            # Stack all predictions for batch processing
            pred_boxes_batch = torch.stack(all_pred_boxes)  # (N, num_queries, 4)
            gt_boxes_batch = torch.stack(all_gt_boxes)  # (N, max_gt, 4)
            obj_scores_batch = torch.stack(all_obj_scores)  # (N, num_queries)

            # Create bbox_mask for the batch
            bbox_mask_batch = torch.zeros(
                gt_boxes_batch.size(0), gt_boxes_batch.size(1)
            )
            for i, gt_box in enumerate(all_gt_boxes):
                valid_count = (gt_box.sum(dim=1) > 0).sum()
                bbox_mask_batch[i, :valid_count] = 1

            bbox_metrics = compute_detr_metrics(
                pred_boxes_batch,
                gt_boxes_batch,
                obj_scores_batch.squeeze(-1),
                bbox_mask_batch,
            )

        # Extract metrics with default values
        avg_val_iou = bbox_metrics.get("avg_iou", 0.0)
        val_map50 = bbox_metrics.get("mAP@0.5", 0.0)
        val_map25 = bbox_metrics.get("mAP@0.25", 0.0)
        recall_iou25 = bbox_metrics.get("recall@0.25", 0.0)

        test_ious.append(avg_val_iou)
        test_map25s.append(val_map25)
        test_recalls_025.append(recall_iou25)  # <--- thêm dòng này

        # Tính recall@0.25 cho train (nếu có bbox)
        if len(train_pred_boxes) > 0:
            pred_boxes_batch = torch.stack(train_pred_boxes)
            gt_boxes_batch = torch.stack(train_gt_boxes)
            obj_scores_batch = torch.stack(train_obj_scores)
            bbox_mask_batch = torch.zeros(
                gt_boxes_batch.size(0), gt_boxes_batch.size(1)
            )
            for i, gt_box in enumerate(train_gt_boxes):
                valid_count = (gt_box.sum(dim=1) > 0).sum()
                bbox_mask_batch[i, :valid_count] = 1
            train_bbox_metrics = compute_detr_metrics(
                pred_boxes_batch,
                gt_boxes_batch,
                obj_scores_batch.squeeze(-1),
                bbox_mask_batch,
            )
            train_recall_025 = train_bbox_metrics.get("recall@0.25", 0.0)
        else:
            train_recall_025 = 0.0
        train_recalls_025.append(train_recall_025)

        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = compute_classification_metrics(
            all_preds, all_labels, all_probs, class_names=None
        )
        metrics["accuracy"] = val_acc * 100

        # Update best epoch info if recall_iou25 improves
        if recall_iou25 > best_recall_iou25:
            best_recall_iou25 = recall_iou25
            best_epoch_info = {
                "epoch": epoch + 1,
                "acc": val_acc,
                "auc": metrics.get("auc"),
                "map25": val_map25,
                "recall_iou25": recall_iou25,
            }

        print_test_metrics(
            metrics, avg_val_iou, val_map50, val_map25, recall_iou25, best_epoch_info
        )
        # Logging
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    datetime.now().isoformat(),
                    epoch + 1,
                    round(train_losses[-1], 6),
                    round(train_accs[-1], 6),
                    round(train_ious[-1], 6),
                    round(test_losses[-1], 6),
                    round(val_acc, 6),
                    round(avg_val_iou, 6),
                    round(val_map50, 6),
                    round(val_map25, 6),
                    round(recall_iou25, 6),
                    optimizer.param_groups[0]["lr"],
                    patience_counter,
                    round(best_acc, 6),
                ]
            )

        # LR scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss / val_total)

        current_lr = optimizer.param_groups[0]["lr"]
        improved = False
        if val_acc > best_acc:
            best_acc = val_acc
            improved = True
        if len(test_losses) > 1 and test_losses[-1] < min(test_losses[:-1]):
            improved = True

        if current_lr < last_lr:
            patience_counter, last_lr = 0, current_lr
        else:
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save top-2 models by accuracy and top-2 by recall_iou25
        if epoch >= 10:
            acc4 = int(round(val_acc * 10000))
            recall_iou254 = int(round(recall_iou25 * 10000))
            weight_name = f"{model_key}_{acc4}_{recall_iou254}.pth"
            weight_path = os.path.join(model_dir, weight_name)

            # Gather all existing models for accuracy ranking
            acc_candidates = []
            for fname in os.listdir(model_dir):
                if (
                    fname.startswith(model_key)
                    and fname.endswith(".pth")
                    and not fname.endswith("_full.pth")
                ):
                    parts = fname.replace(".pth", "").split("_")
                    try:
                        acc_part = parts[-2] if len(parts) > 2 else parts[-1]
                        acc_val = float(acc_part) / 10000
                        fpath = os.path.join(model_dir, fname)
                        acc_candidates.append((acc_val, fpath))
                    except Exception:
                        print(f"Skipping invalid model file: {fname}")
                        continue

            # Gather all existing models for recall ranking
            recall_candidates = []
            for fname in os.listdir(model_dir):
                if (
                    fname.startswith(model_key)
                    and fname.endswith(".pth")
                    and not fname.endswith("_full.pth")
                ):
                    parts = fname.replace(".pth", "").split("_")
                    try:
                        recall_part = parts[-1]
                        recall_val = float(recall_part) / 10000
                        fpath = os.path.join(model_dir, fname)
                        recall_candidates.append((recall_val, fpath))
                    except Exception:
                        continue

            # Sort existing candidates
            acc_candidates = sorted(acc_candidates, key=lambda x: x[0], reverse=True)
            recall_candidates = sorted(
                recall_candidates, key=lambda x: x[0], reverse=True
            )

            # Get top-2 values before adding current
            top2_accs = set(acc for acc, _ in acc_candidates[:2])
            top2_recalls = set(recall for recall, _ in recall_candidates[:2])

            # Check if current model already in top-2 by VALUE (skip if both are already there)
            if val_acc in top2_accs and recall_iou25 in top2_recalls:
                print(
                    f"⏩ Skipped saving {weight_name} (acc {val_acc:.6f} and recall {recall_iou25:.6f} already in top-2)"
                )
            else:
                # Add current model to candidates
                acc_candidates.append((val_acc, weight_path))
                recall_candidates.append((recall_iou25, weight_path))

                # Re-sort and get top-2 paths
                acc_candidates = sorted(
                    acc_candidates, key=lambda x: x[0], reverse=True
                )
                recall_candidates = sorted(
                    recall_candidates, key=lambda x: x[0], reverse=True
                )

                top2_acc_paths = set(path for _, path in acc_candidates[:2])
                top2_recall_paths = set(path for _, path in recall_candidates[:2])
                keep_paths = top2_acc_paths | top2_recall_paths

                # Save if current model is in keep_paths
                if weight_path in keep_paths:
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.state_dict(), weight_path)
                    else:
                        torch.save(model.state_dict(), weight_path)
                    print(
                        f"✅ Saved new model: {weight_name} (acc = {val_acc:.6f}, recall_iou25 = {recall_iou25:.6f})"
                    )

                # Delete models not in keep_paths
                for fname in os.listdir(model_dir):
                    fpath = os.path.join(model_dir, fname)
                    if (
                        fname.startswith(model_key)
                        and fname.endswith(".pth")
                        and not fname.endswith("_full.pth")
                        and fpath not in keep_paths
                    ):
                        try:
                            os.remove(fpath)
                            print(f"🗑️ Deleted model: {fpath}")
                        except Exception as e:
                            print(f"⚠️ Could not delete {fpath}: {e}")
        # Plot metrics every epoch
        # plot_metrics(
        #    train_losses,
        #    train_accs,
        #    test_losses,
        #    test_accs,
        #    os.path.join(plot_dir, f"{model_key}.png"),
        # )

        plot_metrics_det(
            train_losses,
            train_accs,
            test_losses,
            test_accs,
            train_recalls_025,  # <--- truyền recall@0.25 train
            test_recalls_025,  # <--- truyền recall@0.25 test
            test_map25s,
            os.path.join(plot_dir, f"{model_key}_det.png"),
        )

    return model
