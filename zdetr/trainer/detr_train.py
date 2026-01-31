"""
Training script for DETR model
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import torch

from zdetr.data.detr_data import get_detr_dataloaders
from zdetr.data.detr_data_pre import load_detr_metadata
from zdetr.models.detr_model import get_detr_model
from zdetr.trainer.detr_engines import train_detr_model
from zdetr.utils.common import load_config, get_arg_or_config, clear_cuda_memory
from zdetr.utils.detr_common_utils import parse_img_size


def run_detr_train(
    data_folder,
    model_type,
    batch_size,
    num_epochs,
    lr,
    output,
    config_path="config/config.yaml",
    img_size=None,
    patience=50,
    loss_type="ce",
    lambda_bbox=5.0,
    lambda_giou=2.0,
    lambda_obj=1.0,
    num_queries=3,
    max_objects=3,
    pretrained_model_path=None,
    target_column=None,
    sample_viz=False,
):
    """Run DETR training"""
    clear_cuda_memory()

    print("========== DETR Hyperparameters ==========")
    print(f"Model type      : {model_type}")
    print(f"Num queries     : {num_queries}")
    print(f"Max objects     : {max_objects}")
    print(f"Lambda bbox     : {lambda_bbox}")
    print(f"Lambda GIoU     : {lambda_giou}")
    print(f"Lambda obj      : {lambda_obj}")
    print(f"Batch size      : {batch_size}")
    print(f"Num epochs      : {num_epochs}")
    print(f"Learning rate   : {lr}")
    print(f"Patience        : {patience}")
    print(f"Loss type       : {loss_type}")
    print(f"Image size      : {img_size}")
    print("==========================================")

    # Load metadata
    train_df, test_df, class_names = load_detr_metadata(
        data_folder, config_path, target_column=target_column
    )

    # Get dataloaders
    train_loader, test_loader = get_detr_dataloaders(
        train_df,
        test_df,
        data_folder,
        batch_size=batch_size,
        config_path=config_path,
        img_size=img_size,
        mode="train",
        max_objects=max_objects,
    )

    # Get model
    model = get_detr_model(
        model_type=model_type, num_classes=len(class_names), num_queries=num_queries
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if pretrained_model_path:
        try:
            model.load_state_dict(
                torch.load(pretrained_model_path, map_location=device), strict=False
            )
            print(f"✅ Loaded pretrained model from {pretrained_model_path}")
        except Exception as e:
            print(f"⚠️ Error loading pretrained: {e}. Training from scratch.")

    model = model.to(device)

    # Sample visualization
    if sample_viz:
        from src.trainer.train_m2 import sample_viz_batches

        viz_dir = os.path.join(output, "train_sample_detr")
        print(f"🔍 Visualizing training batches to {viz_dir}...")
        sample_viz_batches(train_loader, viz_dir, class_names, num_batches=5)

    # Train
    trained_model = train_detr_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        model_name=model_type,
        output=output,
        dataset_folder=data_folder,
        train_df=train_df,
        patience=patience,
        loss_type=loss_type,
        lambda_bbox=lambda_bbox,
        lambda_giou=lambda_giou,
        lambda_obj=lambda_obj,
    )

    return trained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DETR Model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--img_size", type=str, default=None)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument(
        "--loss_type", type=str, choices=["ce", "focal", "ldam"], default=None
    )
    parser.add_argument("--lambda_bbox", type=float, default=5.0)
    parser.add_argument("--lambda_giou", type=float, default=2.0)
    parser.add_argument("--lambda_obj", type=float, default=1.0)
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument("--max_objects", type=int, default=3)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--target_column", type=str, default=None)
    parser.add_argument("--sample_viz", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)

    data_folder = get_arg_or_config(args.data_folder, config.get("data_folder"), None)
    model_type = get_arg_or_config(args.model_type, config.get("model_type"), None)
    batch_size = get_arg_or_config(args.batch_size, config.get("batch_size"), 16)
    num_epochs = get_arg_or_config(args.num_epochs, config.get("num_epochs"), 100)
    patience = get_arg_or_config(args.patience, config.get("patience"), 150)
    lr = get_arg_or_config(args.lr, config.get("lr"), 1e-4)
    output = get_arg_or_config(args.output, config.get("output"), "output")
    img_size = get_arg_or_config(args.img_size, config.get("image_size"), None)
    loss_type = (
        args.loss_type
        if args.loss_type is not None
        else config.get("loss_type", "focal")
    )

    if img_size and isinstance(img_size, str):
        img_size = parse_img_size(img_size)

    run_detr_train(
        data_folder=data_folder,
        model_type=model_type,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        output=output,
        config_path=args.config,
        img_size=img_size,
        patience=patience,
        loss_type=loss_type,
        lambda_bbox=args.lambda_bbox,
        lambda_giou=args.lambda_giou,
        lambda_obj=args.lambda_obj,
        num_queries=args.num_queries,
        max_objects=args.max_objects,
        pretrained_model_path=args.pretrained_model_path,
        target_column=args.target_column,
        sample_viz=args.sample_viz,
    )
