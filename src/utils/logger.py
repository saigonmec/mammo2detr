"""Logging: console/file logs, CSV/JSONL history, optional TensorBoard / W&B."""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(output_dir: str | Path | None, rank: int = 0, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level if rank == 0 else logging.WARNING)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname).1s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if output_dir is not None and rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(output_dir) / "log.txt")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    for noisy in ("PIL", "matplotlib", "timm", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("src")


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields: list[str] | None = None
        if self.path.exists():
            with open(self.path) as f:
                self.fields = next(csv.reader(f), None)

    def log(self, row: dict[str, Any]) -> None:
        row = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
        if self.fields is None or set(row) - set(self.fields):
            self.fields = sorted(set(self.fields or []) | set(row), key=lambda k: (k not in ("epoch", "step"), k))
            existing = []
            if self.path.exists():
                with open(self.path) as f:
                    existing = list(csv.DictReader(f))
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fields)
                w.writeheader()
                for e in existing:
                    w.writerow(e)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow({k: row.get(k, "") for k in self.fields})


class ExperimentLogger:
    """Fan-out to CSV + JSONL (+ TensorBoard / W&B when enabled and installed)."""

    def __init__(self, output_dir: str | Path, tensorboard: bool = False, wandb: bool = False, wandb_project: str = "", run_name: str | None = None, config: dict | None = None):
        self.output_dir = Path(output_dir)
        self.csv = CSVLogger(self.output_dir / "history.csv")
        self.tb = None
        self.wb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(self.output_dir / "tb"))
            except Exception as e:  # pragma: no cover
                logging.getLogger(__name__).warning("TensorBoard unavailable: %s", e)
        if wandb:
            try:
                import wandb as _wandb

                self.wb = _wandb.init(project=wandb_project, name=run_name, config=config, dir=str(self.output_dir))
            except Exception as e:  # pragma: no cover
                logging.getLogger(__name__).warning("W&B unavailable: %s", e)

    def log_scalars(self, scalars: dict[str, Any], step: int, prefix: str = "") -> None:
        flat = {f"{prefix}{k}": float(v) for k, v in scalars.items() if isinstance(v, (int, float))}
        if self.tb is not None:
            for k, v in flat.items():
                self.tb.add_scalar(k, v, step)
        if self.wb is not None:
            self.wb.log(flat, step=step)

    def log_epoch(self, row: dict[str, Any]) -> None:
        self.csv.log(row)
        with open(self.output_dir / "history.jsonl", "a") as f:
            f.write(json.dumps({k: v for k, v in row.items() if isinstance(v, (int, float, str))}) + "\n")

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.wb is not None:
            self.wb.finish()
