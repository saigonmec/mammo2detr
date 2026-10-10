"""Measure parameters, forward/backward time and memory for several backbones.

    python -m src.tools.benchmark_backbones --backbones resnet50 convnextv2_tiny efficientnetv2s --size 640 640
"""
from __future__ import annotations

import argparse
import time

import torch

from ..config import Config
from ..models import build_model


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbones", nargs="+", default=["resnet50", "convnextv2_tiny", "efficientnetv2s"])
    p.add_argument("--size", type=int, nargs=2, default=[640, 640])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--strides", type=int, nargs="+", default=[32])
    p.add_argument("--type", default=None, help="standard|deformable (auto from strides)")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args(argv)
    from ..utils.misc import resolve_device

    device = resolve_device(args.device)
    ttype = args.type or ("deformable" if len(args.strides) > 1 else "standard")
    print(f"{'backbone':24s} {'params(M)':>10s} {'fwd ms':>8s} {'fwd+bwd ms':>11s} {'mem GB':>7s}")
    for name in args.backbones:
        try:
            cfg = Config.load(None, [f"model.backbone.name={name}", f"model.backbone.pretrained={str(args.pretrained).lower()}", f"model.backbone.out_strides={list(args.strides)}", f"model.transformer.type={ttype}", f"data.img_size={list(args.size)}"])
            model = build_model(cfg).to(device)
            x = torch.randn(args.batch_size, 3, *args.size, device=device)
            tg = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device), "labels": torch.zeros(1, dtype=torch.long, device=device), "image_labels": {"cancer": torch.tensor(1, device=device)}} for _ in range(args.batch_size)]
            model.eval()
            with torch.no_grad():
                model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(args.iters):
                    model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                fwd = (time.time() - t0) / args.iters * 1000
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(args.iters):
                out = model(x, None, tg)
                loss = out["pred_boxes"].sum() + out["pred_logits"].sum()
                loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            fb = (time.time() - t0) / args.iters * 1000
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"{name:24s} {n_params:10.1f} {fwd:8.0f} {fb:11.0f} {mem:7.1f}")
            del model, out, loss
        except Exception as e:  # noqa: BLE001
            print(f"{name:24s} ERROR: {e}")


if __name__ == "__main__":
    main()
