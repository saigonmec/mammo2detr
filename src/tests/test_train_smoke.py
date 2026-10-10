"""End-to-end: train 1 epoch on synthetic data (CPU), then evaluate and predict through the CLIs."""
import json
from pathlib import Path

import pytest

from src import evaluate as evaluate_cli
from src import predict as predict_cli
from src import train as train_cli

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_train_eval_predict(synthetic_root, tmp_path):
    out_dir = tmp_path / "runs"
    summary = train_cli.main([
        "--config", str(ROOT / "configs" / "debug_cpu.yaml"),
        "--data-root", str(synthetic_root),
        "--output-dir", str(out_dir),
        "--run-name", "smoke",
        "--set", "optim.epochs=1",
        "--set", "data.img_size=[64,64]",
        "--set", "eval.viz_samples=2",
    ])
    run = out_dir / "smoke"
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    assert (run / "history.csv").exists() and (run / "config.yaml").exists()
    assert "test" in summary and "det/froc_mean" in summary["test"]
    metrics = json.loads((run / "test" / "metrics.json").read_text())
    assert "cls/cancer/auc" in metrics and (run / "test" / "predictions_images.csv").exists()

    m = evaluate_cli.main(["--checkpoint", str(run / "best.pt"), "--split", "val", "--viz", "1", "--output-dir", str(tmp_path / "ev")])
    assert "det/ap50" in m and (tmp_path / "ev" / "metrics.json").exists()

    predict_cli.main(["--checkpoint", str(run / "best.pt"), "--input", str(synthetic_root / "images"), "--output", str(tmp_path / "pred"), "--attention", "--score-threshold", "0.0"])
    assert (tmp_path / "pred" / "predictions.csv").exists()
    assert len(list((tmp_path / "pred" / "overlays").glob("*.jpg"))) == 24
    assert len(list((tmp_path / "pred" / "attention").glob("*.jpg"))) == 24

    # resume for one more epoch
    train_cli.main(["--config", str(ROOT / "configs" / "debug_cpu.yaml"), "--data-root", str(synthetic_root), "--output-dir", str(out_dir), "--run-name", "smoke", "--resume", str(run / "last.pt"), "--set", "optim.epochs=2", "--set", "data.img_size=[64,64]"])
    rows = (run / "history.csv").read_text().strip().splitlines()
    assert len(rows) == 3  # header + 2 epochs
