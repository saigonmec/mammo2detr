"""Typed configuration for :mod:`src`.

A run is fully described by one YAML file (see ``src/configs``) that is parsed
into nested dataclasses with *strict* key checking, plus optional dotted CLI
overrides (``--set optim.lr=2e-4 --set data.img_size=[640,512]``).
"""
from __future__ import annotations

import copy
import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class ImageTargetConfig:
    """One image-level classification target (e.g. ``cancer``, ``density``)."""

    name: str = "cancer"
    num_classes: int = 2
    loss: str = "ce"  # ce | focal | ldam
    weight: float = 1.0
    class_weights: str | list[float] | None = "balanced"  # balanced | sqrt | null | explicit list
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    positive_index: int = 1  # index of the "positive" class for sensitivity/AUC


@dataclass
class AugConfig:
    hflip: float = 0.5
    vflip: float = 0.5
    rotate90: float = 0.0
    affine_p: float = 0.5
    affine_scale: tuple[float, float] = (0.9, 1.1)
    affine_translate: float = 0.05
    affine_rotate: float = 10.0
    affine_shear: float = 5.0
    bbox_safe_crop_p: float = 0.3
    bbox_safe_crop_scale: float = 0.8  # crop side as fraction of the (resized) image
    brightness_contrast_p: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    gamma_p: float = 0.3
    clahe_p: float = 0.3
    blur_sharpen_p: float = 0.2
    noise_p: float = 0.15
    min_visibility: float = 0.3


@dataclass
class DataConfig:
    root: str = ""
    metadata_csv: str = "metadata.csv"
    image_col: str | None = None  # auto: link | image_path | filename
    image_id_col: str = "image_id"
    patient_col: str | None = "patient_id"
    split_col: str = "split"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    val_fraction: float = 0.1  # carve a patient-level val split from train when none exists
    image_labels: list[ImageTargetConfig] = field(default_factory=lambda: [ImageTargetConfig()])
    bbox_cols: tuple[str, str, str, str] = ("x", "y", "width", "height")
    bbox_format: str = "xywh"  # xywh (COCO) | xyxy
    lesion_label_col: str | None = None  # column with lesion category; None -> single class
    det_class_names: list[str] = field(default_factory=lambda: ["lesion"])
    img_width_col: str = "img_width"
    img_height_col: str = "img_height"
    img_size: tuple[int, int] = (448, 448)  # (H, W) network input
    resize_mode: str = "stretch"  # stretch | letterbox
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    min_box_area: float = 4.0  # pixels^2 in the original image
    max_boxes: int | None = None
    aug: AugConfig = field(default_factory=AugConfig)
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    balanced_sampler: bool = True
    balanced_sampler_target: str | None = None  # defaults to the first image label
    drop_last: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.img_size, int):
            self.img_size = (self.img_size, self.img_size)
        self.img_size = tuple(int(v) for v in self.img_size)  # type: ignore[assignment]
        if len(self.img_size) != 2:
            raise ConfigError("data.img_size must be an int or [H, W]")
        self.mean = tuple(float(v) for v in self.mean)  # type: ignore[assignment]
        self.std = tuple(float(v) for v in self.std)  # type: ignore[assignment]
        if self.resize_mode not in ("stretch", "letterbox"):
            raise ConfigError("data.resize_mode must be 'stretch' or 'letterbox'")
        if self.bbox_format not in ("xywh", "xyxy"):
            raise ConfigError("data.bbox_format must be 'xywh' or 'xyxy'")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class BackboneConfig:
    name: str = "resnet50"  # registry alias or raw timm name ("timm:<name>")
    pretrained: bool = True
    out_strides: list[int] = field(default_factory=lambda: [8, 16, 32])  # feature levels fed to the transformer
    freeze_at: int = 0  # freeze the first N feature stages (0 = none)
    freeze_bn: bool = False  # convert BatchNorm to FrozenBatchNorm (DETR default for small batches)
    unfreeze_last_blocks: int | None = None  # ViT: train only the last N blocks (None = all)
    drop_path_rate: float = 0.0
    timm_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformerConfig:
    type: str = "deformable"  # standard | deformable
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "relu"
    num_queries: int = 20
    enc_n_points: int = 4  # deformable only
    dec_n_points: int = 4  # deformable only
    box_refine: bool = True  # iterative box refinement across decoder layers
    aux_loss: bool = True
    learn_anchors: bool = True  # learnable 4-D anchor boxes as positional queries (DAB-style)
    pos_temperature: float = 20.0  # sine temperature for memory and anchor embeddings (DAB/DINO use 20)


@dataclass
class DenoisingConfig:
    enabled: bool = True
    num_groups: int = 5
    box_noise_scale: float = 0.4
    label_noise_ratio: float = 0.2
    contrastive: bool = True  # DINO-style positive + negative noised queries


@dataclass
class HeadsConfig:
    image_cls: str = "hybrid"  # query | pool | hybrid | none
    cls_hidden_dim: int = 256
    cls_dropout: float = 0.1


@dataclass
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    denoising: DenoisingConfig = field(default_factory=DenoisingConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)


# --------------------------------------------------------------------------- #
# Loss / optimisation / evaluation / logging
# --------------------------------------------------------------------------- #
@dataclass
class LossConfig:
    det_class_weight: float = 2.0
    bbox_weight: float = 5.0
    giou_weight: float = 2.0
    matcher_cost_class: float = 2.0
    matcher_cost_bbox: float = 5.0
    matcher_cost_giou: float = 2.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    image_loss_weight: float = 1.0
    aux_weight: float = 1.0
    dn_weight: float = 1.0


@dataclass
class EMAConfig:
    enabled: bool = True
    decay: float = 0.9998
    warmup_steps: int = 500


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 0.1
    epochs: int = 50
    scheduler: str = "cosine"  # cosine | step | multistep | plateau | none
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.01
    step_milestones: list[int] = field(default_factory=lambda: [40])
    step_gamma: float = 0.1
    plateau_patience: int = 10
    amp: bool = True
    accumulate_steps: int = 1
    ema: EMAConfig = field(default_factory=EMAConfig)
    early_stopping_patience: int | None = None


@dataclass
class EvalConfig:
    score_threshold: float = 0.05
    topk: int = 10
    nms_iou: float | None = None  # optional NMS on predictions (DETR normally needs none)
    ap_iou_thresholds: list[float] = field(default_factory=lambda: [0.25, 0.5])
    froc_fppi: list[float] = field(default_factory=lambda: [0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    froc_match: str = "iou"  # iou | center
    froc_iou: float = 0.25
    monitor: str = "det/froc_mean"  # metric used for best-checkpoint selection
    monitor_mode: str = "max"
    eval_every: int = 1
    eval_on_test_at_end: bool = True
    viz_samples: int = 8


@dataclass
class LoggingConfig:
    output_dir: str = "runs"
    run_name: str | None = None
    log_every: int = 20
    save_top_k: int = 2
    save_last: bool = True
    tensorboard: bool = False
    wandb: bool = False
    wandb_project: str = "mammo2detr"


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    deterministic: bool = False
    resume: str | None = None
    init_weights: str | None = None
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------ #
    @property
    def num_feature_levels(self) -> int:
        return len(self.model.backbone.out_strides)

    def validate(self) -> "Config":
        t = self.model.transformer
        if t.d_model % t.nhead != 0:
            raise ConfigError("transformer.d_model must be divisible by nhead")
        if t.type not in ("standard", "deformable"):
            raise ConfigError("transformer.type must be 'standard' or 'deformable'")
        if t.type == "standard" and self.num_feature_levels != 1:
            raise ConfigError(
                "transformer.type='standard' uses exactly one feature level; "
                "set backbone.out_strides to a single stride (e.g. [32] or [16]) "
                "or use transformer.type='deformable' for multi-scale."
            )
        if not self.data.image_labels and self.model.heads.image_cls != "none":
            raise ConfigError("data.image_labels is empty but heads.image_cls != 'none'")
        if self.model.heads.image_cls not in ("query", "pool", "hybrid", "none"):
            raise ConfigError("heads.image_cls must be query|pool|hybrid|none")
        if self.eval.monitor_mode not in ("max", "min"):
            raise ConfigError("eval.monitor_mode must be max|min")
        names = [t_.name for t_ in self.data.image_labels]
        if len(set(names)) != len(names):
            raise ConfigError("data.image_labels names must be unique")
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        return _from_dict(cls, d).validate()

    @classmethod
    def load(cls, path: str | Path | None, overrides: list[str] | None = None) -> "Config":
        d: dict[str, Any] = {}
        if path is not None:
            with open(path) as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ConfigError(f"{path}: top level must be a mapping")
            # Support legacy files that nest everything under a top-level 'config:' key.
            d = loaded["config"] if set(loaded.keys()) == {"config"} else loaded
        d = apply_overrides(copy.deepcopy(d), overrides or [])
        return cls.from_dict(d)


# --------------------------------------------------------------------------- #
# dict <-> dataclass plumbing
# --------------------------------------------------------------------------- #
def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _is_union(tp: Any) -> bool:
    return typing.get_origin(tp) is typing.Union or isinstance(tp, types.UnionType)


def _coerce_scalar(tp: Any, value: Any, path: str) -> Any:
    """Coerce YAML scalars to the annotated type (``yaml.safe_load("2e-4")`` is a *string*)."""
    if value is None or isinstance(value, bool):
        return value
    if tp is float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as e:
                raise ConfigError(f"{path}: expected a float, got '{value}'") from e
    if tp is int:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                f = float(value)
                if f.is_integer():
                    return int(f)
            except ValueError:
                pass
            raise ConfigError(f"{path}: expected an int, got '{value}'")
    if tp is bool and isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "on", "1"):
            return True
        if low in ("false", "no", "off", "0"):
            return False
        raise ConfigError(f"{path}: expected a bool, got '{value}'")
    return value


def _coerce(tp: Any, value: Any, path: str) -> Any:
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if value is None:
            return tp()
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping for {tp.__name__}, got {type(value).__name__}")
        return _from_dict(tp, value, path + ".")
    if _is_union(tp):
        if value is None:
            return None
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        for a in args:
            if dataclasses.is_dataclass(a) and isinstance(value, dict):
                return _coerce(a, value, path)
        for a in args:
            origin = typing.get_origin(a)
            if origin in (list, tuple) and isinstance(value, (list, tuple)):
                return _coerce(a, value, path)
        for a in args:
            if a in (float, int, bool) and isinstance(value, (str, int, float)):
                try:
                    return _coerce_scalar(a, value, path)
                except ConfigError:
                    continue
        return value
    origin = typing.get_origin(tp)
    if origin in (list,):
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list")
        args = typing.get_args(tp)
        if args and dataclasses.is_dataclass(args[0]):
            return [_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(value)]
        return list(value)
    if origin in (tuple,):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return value  # e.g. img_size: 448 -> handled in __post_init__
        return tuple(value)
    return _coerce_scalar(tp, value, path)


def _from_dict(cls: type, d: dict[str, Any], path: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in d.items():
        if key not in fields:
            raise ConfigError(
                f"Unknown config key '{path}{key}' for {cls.__name__}. "
                f"Valid keys: {sorted(fields)}"
            )
        kwargs[key] = _coerce(hints[key], value, path + key)
    try:
        return cls(**kwargs)
    except TypeError as e:  # pragma: no cover - defensive
        raise ConfigError(f"{path or cls.__name__}: {e}") from e


def apply_overrides(d: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (values parsed as YAML) to a nested dict."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override '{item}' must look like key.path=value")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw) if raw != "" else None
        node = d
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node or node[p] is None:
                node[p] = {}
            if not isinstance(node[p], dict):
                raise ConfigError(f"Override '{item}': '{p}' is not a mapping")
            node = node[p]
        node[parts[-1]] = value
    return d
