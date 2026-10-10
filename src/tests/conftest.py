import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    from src.tools.make_synthetic_dataset import main as make

    out = tmp_path_factory.mktemp("synth")
    make(["--out", str(out), "--n", "24", "--size", "96", "80"])
    return out


@pytest.fixture()
def tiny_cfg():
    from src.config import Config

    return Config.load(
        None,
        [
            "model.backbone.name=timm:resnet18",
            "model.backbone.pretrained=false",
            "model.backbone.out_strides=[32]",
            "model.transformer.type=standard",
            "data.img_size=[64,80]",
            "model.transformer.d_model=64",
            "model.transformer.nhead=4",
            "model.transformer.dim_feedforward=128",
            "model.transformer.num_encoder_layers=1",
            "model.transformer.num_decoder_layers=2",
            "model.transformer.num_queries=5",
            "model.denoising.num_groups=2",
        ],
    )
