import pytest

from src.config import Config, ConfigError


def test_defaults_validate():
    cfg = Config.load(None)
    assert cfg.data.img_size == (448, 448)
    assert cfg.num_feature_levels == 3 and cfg.model.transformer.type == "deformable"


def test_overrides_and_nested_lists():
    cfg = Config.load(None, ["optim.lr=2e-4", "data.img_size=[640,512]", "data.image_labels=[{name: cancer, num_classes: 2}, {name: density, num_classes: 4}]"])
    assert cfg.optim.lr == 2e-4
    assert cfg.data.img_size == (640, 512)
    assert [t.name for t in cfg.data.image_labels] == ["cancer", "density"]


def test_strict_keys_and_validation():
    with pytest.raises(ConfigError):
        Config.load(None, ["optim.lrr=1"])
    with pytest.raises(ConfigError):
        Config.load(None, ["model.backbone.out_strides=[16,32]", "model.transformer.type=standard"])  # standard needs one level
    Config.load(None, ["model.backbone.out_strides=[16,32]", "model.transformer.type=deformable"])
    assert Config.load(None).model.transformer.type == "deformable"


def test_roundtrip_save_load(tmp_path):
    cfg = Config.load(None, ["optim.epochs=3"])
    p = tmp_path / "c.yaml"
    cfg.save(p)
    cfg2 = Config.load(p)
    assert cfg2.to_dict() == cfg.to_dict()
