"""mammo2detr ``src`` package.

A standardized, multi-task DETR-style framework for mammography:
image-level classification (one or more targets) + lesion detection,
with pluggable backbones (timm), standard or multi-scale deformable
transformers, denoising training and mammography-oriented metrics (FROC).

Entry points::

    python -m src.train    --config src/configs/resnet50_448.yaml --set data.root=/path/to/data
    python -m src.evaluate --checkpoint runs/<run>/best.pt --split test
    python -m src.predict  --checkpoint runs/<run>/best.pt --input /folder --output /out
"""

__version__ = "1.0.0"
