import torch

from src.config import Config
from src.losses import build_criterion
from src.models import build_model


def _targets(bs, device="cpu"):
    tg = []
    for i in range(bs):
        n = (i % 3)
        b = torch.rand(n, 4) * 0.4 + 0.3
        b[:, 2:] = b[:, 2:] * 0.3 + 0.05
        tg.append({"boxes": b, "labels": torch.zeros(n, dtype=torch.long), "image_labels": {"cancer": torch.tensor(int(n > 0))}})
    return tg


def _run(cfg, bs=2, size=(64, 80)):
    model = build_model(cfg)
    crit = build_criterion(cfg, {"cancer": [5, 5]})
    x = torch.randn(bs, 3, *size)
    m = torch.zeros(bs, *size, dtype=torch.bool)
    m[:, :, -8:] = True
    tg = _targets(bs)
    model.train()
    out = model(x, m, tg)
    losses = crit(out, tg)
    losses["loss_total"].backward()
    assert torch.isfinite(losses["loss_total"])
    assert out["pred_logits"].shape == (bs, cfg.model.transformer.num_queries, 1)
    assert out["pred_boxes"].shape == (bs, cfg.model.transformer.num_queries, 4)
    assert len(out["aux_outputs"]) == cfg.model.transformer.num_decoder_layers - 1
    model.eval()
    with torch.no_grad():
        o = model(x, m)
    assert "dn_outputs" not in o and o["pred_boxes"].min() >= 0 and o["pred_boxes"].max() <= 1
    return model, out, losses


def test_standard(tiny_cfg):
    _, out, losses = _run(tiny_cfg)
    assert out["dn_outputs"] is not None and "loss_bbox_dn" in losses
    assert out["image_logits"]["cancer"].shape == (2, 2)


def test_deformable_multiscale():
    cfg = Config.load(None, ["model.backbone.name=timm:resnet18", "model.backbone.pretrained=false", "model.backbone.out_strides=[8,16,32]", "model.transformer.type=deformable", "model.transformer.d_model=64", "model.transformer.nhead=4", "model.transformer.dim_feedforward=128", "model.transformer.num_encoder_layers=1", "model.transformer.num_decoder_layers=2", "model.transformer.num_queries=5", "model.heads.image_cls=query"])
    _run(cfg)


def test_vit_backbone_pyramid():
    cfg = Config.load(None, ["model.backbone.name=timm:vit_small_patch16_224", "model.backbone.pretrained=false", "model.backbone.out_strides=[8,16,32]", "model.backbone.unfreeze_last_blocks=1", "model.transformer.type=deformable", "model.transformer.d_model=64", "model.transformer.nhead=4", "model.transformer.dim_feedforward=128", "model.transformer.num_encoder_layers=0", "model.transformer.num_decoder_layers=2", "model.transformer.num_queries=5", "model.heads.image_cls=pool"])
    model, _, _ = _run(cfg, size=(64, 96))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.model.")}
    assert all(("blocks.11" in n) or n.startswith("backbone.model.norm") for n in trainable)


def test_no_dn_no_refine(tiny_cfg):
    cfg = Config.from_dict({**tiny_cfg.to_dict(), "model": {**tiny_cfg.to_dict()["model"], "denoising": {"enabled": False}, "transformer": {**tiny_cfg.to_dict()["model"]["transformer"], "box_refine": False}}})
    _, out, losses = _run(cfg)
    assert "dn_outputs" not in out


def test_attention_storage(tiny_cfg):
    model = build_model(tiny_cfg).eval()
    model.set_store_attention(True)
    x = torch.randn(1, 3, 64, 80)
    with torch.no_grad():
        out = model(x)
    attn = model.decoder_layers[-1].last_attn
    h, w = out["spatial_shapes"][0]
    assert attn.shape == (1, tiny_cfg.model.transformer.num_queries + 1, h * w)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-4)


def test_attention_storage_deformable():
    cfg = Config.load(None, ["model.backbone.name=timm:resnet18", "model.backbone.pretrained=false", "model.transformer.d_model=64", "model.transformer.nhead=4", "model.transformer.dim_feedforward=128", "model.transformer.num_encoder_layers=1", "model.transformer.num_decoder_layers=2", "model.transformer.num_queries=5"])
    model = build_model(cfg).eval()
    model.set_store_attention(True)
    with torch.no_grad():
        model(torch.randn(1, 3, 64, 80))
    locs, weights = model.decoder_layers[-1].last_attn
    assert locs.shape[:2] == (1, 6) and locs.shape[-1] == 2 and weights.shape == locs.shape[:-1]
    assert torch.allclose(weights.sum(-1).sum(-1), torch.ones(1, 6, 4), atol=1e-4)
