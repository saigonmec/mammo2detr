# mammo2detr / `src` — multi-task DETR for mammography

A clean, standardized re-implementation of the `zdetr` prototype:
**image-level classification (one or more targets) + lesion detection** with any `timm`
backbone, a standard or multi-scale deformable transformer, DAB-style anchors with iterative
refinement, contrastive denoising training, and mammography metrics (FROC, AP, AUC).
See [`docs/ANALYSIS_zdetr.md`](../docs/ANALYSIS_zdetr.md) for the analysis of the old code
and the reasoning behind every design decision (in Vietnamese).

## Install

```bash
pip install -r src/requirements.txt          # torch/timm/albumentations/opencv/...
pip install pytest && (cd src && python -m pytest -q)   # optional: run the test suite
```

## Data format

One `metadata.csv` in the data root, **one row per annotation** (images without lesions have
one row with empty box columns):

| column | required | meaning |
|---|---|---|
| `image_id` | yes | unique image identifier |
| `link` / `image_path` / `path` | yes | image path relative to the data root (or absolute) |
| `split` | yes | `train` / `val` / `test` (if `val` is missing, a patient-level val split is carved from train) |
| `cancer` (configurable) | yes | image-level label(s); numeric or string classes |
| `x, y, width, height` | optional | COCO pixel box in the original image (`data.bbox_format: xyxy` also supported) |
| `patient_id` | recommended | used for leakage-free splits |
| `img_width, img_height` | optional | used to validate boxes without opening images |
| lesion category column | optional | `data.lesion_label_col` → multi-class detection |

Check a dataset before training:

```bash
python -m src.tools.validate_data --data-root /data/mammo --check-images sample
python -m src.tools.make_splits --csv /data/mammo/metadata.csv --out /data/mammo/metadata_split.csv --val 0.1 --test 0.2
```

## Train / evaluate / predict

```bash
# ResNet-50, 448x448, single-scale transformer (fast baseline)
python -m src.train --config src/configs/resnet50_448.yaml --data-root /data/mammo --run-name r50_448

# ConvNeXt-V2 tiny, 640x640, multi-scale deformable (strides 8/16/32)
python -m src.train --config src/configs/convnextv2_tiny_640.yaml --data-root /data/mammo

# DINOv3 ViT-S/16 with a ViTDet-style feature pyramid, last 4 blocks fine-tuned
python -m src.train --config src/configs/dinov3_vits16_512.yaml --data-root /data/mammo

# any key can be overridden from the command line (quote list values: zsh/bash treat [..] as a glob)
python -m src.train --config src/configs/base.yaml --data-root /data/mammo \
    --set model.backbone.name=efficientnetv2s --set 'data.img_size=[1024,768]' \
    --set data.resize_mode=letterbox --set optim.epochs=80 --set eval.monitor=cls/cancer/auc

# multi-GPU
torchrun --nproc_per_node=4 -m src.train --config src/configs/convnextv2_tiny_640.yaml --data-root /data/mammo

# resume / evaluate / inference on a folder (+ decoder attention overlays)
python -m src.train --config ... --resume runs/<run>/last.pt
python -m src.evaluate --checkpoint runs/<run>/best.pt --split test --data-root /data/mammo
python -m src.predict  --checkpoint runs/<run>/best.pt --input /folder/of/images --output /out --attention
```

Every run directory contains `config.yaml`, `log.txt`, `history.csv/.jsonl`, `last.pt`,
`best.pt`, `checkpoints/` (top-k by `eval.monitor`), `viz/` (GT in green, predictions coloured
by score) and, after training, `test/metrics.json` + `test/predictions_*.csv`.

## Backbones

Aliases keep the names used in `zdetr` (`resnet50`, `convnextv2_tiny`, `efficientnetv2s`,
`maxvit_tiny`, `swinv2_tiny`, `dinov2_small`, `dinov3_vit16small`, `eva02_small`, …) and map to
`timm` ids in `src/models/backbones.py`; any other timm model works with `timm:<name>`.
CNN / hierarchical models expose strides 4-32 (`model.backbone.out_strides`), plain ViTs
expose their patch stride and a ViTDet-style pyramid (`[8,16,32]` for patch 16).

| setting | effect |
|---|---|
| `transformer.type: deformable` (default) | multi-scale deformable attention (pure PyTorch) over `out_strides`; converged fastest in our checks |
| `transformer.type: standard` | dense attention on one level (`out_strides: [16]`); conditional cross-attention with a positional-kernel init, anchors, refinement |
| `transformer.pos_temperature` | sine temperature (20 as in DAB-DETR/DINO; 10000 = original DETR, noticeably slower) |
| `denoising.*` | contrastive denoising queries with a proper attention mask |
| `heads.image_cls: query/pool/hybrid` | image-level head from a [CLS]-like query, pooled memory, or both |
| `data.image_labels` | list of targets, each with its own loss / class weights (multi-task) |
| `eval.monitor` | metric used to select checkpoints (`det/froc_mean`, `cls/cancer/auc`, `loss`, …) |

## Metrics

* Classification per target: accuracy, balanced accuracy, AUC, AP, sensitivity, specificity, F1,
  Youden-optimal threshold, per-class recall, confusion matrix.
* Detection: `ap25`, `ap50`, `map_coco` (0.5:0.95), **FROC** `sens@fppi{0.125,…,4}` and
  `froc_mean` (false positives counted on *all* images, matching by IoU or centre point),
  `img_recall@k` / `lesion_recall@k`, `box_img_auc` (max box score as an image classifier).

## What was validated

* 30 pytest tests (`cd src && python -m pytest`): box ops vs torchvision, matcher, criterion, DN
  mask semantics, model forward for standard/deformable/ViT backbones, hand-computed AP/FROC/AUC,
  dataset + loader, and an end-to-end train → evaluate → predict → resume run on synthetic data.
* A 40-epoch CPU run on synthetic data (160 images, 112x96) through the CLI: val FROC-mean
  0.02 → 0.43, image recall@1 0 → 0.63, classification AUC → 1.0; see
  `docs/ANALYSIS_zdetr.md` section 5 for the numbers and for the standard-vs-deformable comparison.
* `torchrun --nproc_per_node=2` on CPU (gloo) runs the same training loop (DDP + sharded balanced sampler).

## Layout

```
src/
  config.py            typed config (strict keys, CLI overrides)
  train.py evaluate.py predict.py
  data/                metadata, transforms, dataset, samplers, loader builder
  models/              backbones, attention, transformer, denoising, heads, detr
  losses/              matcher, focal/LDAM, set criterion
  engine/              trainer, evaluator, postprocess, metrics
  utils/               box ops, misc/dist, checkpoint, ema, scheduler, logger
  viz/                 drawing + heat-map overlays
  tools/               validate_data, make_splits, make_synthetic_dataset, benchmark_backbones
  configs/             base + ready-to-run presets
  tests/               pytest suite (unit + end-to-end on synthetic data)
```
