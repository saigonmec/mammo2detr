import torch

from src.config import Config
from src.losses import HungarianMatcher, build_criterion


def test_matcher_assigns_nearest_queries():
    gt = torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]])
    preds = torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.71, 0.69, 0.2, 0.2], [0.21, 0.2, 0.1, 0.1]])
    logits = torch.full((1, 3, 1), 2.0)
    out = {"pred_logits": logits, "pred_boxes": preds[None]}
    idx = HungarianMatcher()(out, [{"labels": torch.zeros(2, dtype=torch.long), "boxes": gt}])
    src, tgt = idx[0]
    pairs = dict(zip(tgt.tolist(), src.tolist()))
    assert pairs == {0: 2, 1: 1}


def test_matcher_empty_targets():
    out = {"pred_logits": torch.randn(2, 4, 1), "pred_boxes": torch.rand(2, 4, 4)}
    idx = HungarianMatcher()(out, [{"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}] * 2)
    assert all(len(i) == 0 and len(j) == 0 for i, j in idx)


def test_criterion_prefers_correct_predictions():
    cfg = Config.load(None)
    crit = build_criterion(cfg, {"cancer": [10, 10]})
    gt = torch.tensor([[0.3, 0.3, 0.2, 0.2]])
    targets = [{"labels": torch.zeros(1, dtype=torch.long), "boxes": gt, "image_labels": {"cancer": torch.tensor(1)}}]
    good = {"pred_logits": torch.tensor([[[4.0], [-4.0], [-4.0]]]), "pred_boxes": torch.tensor([[[0.3, 0.3, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]]]), "image_logits": {"cancer": torch.tensor([[-3.0, 3.0]])}}
    bad = {"pred_logits": torch.tensor([[[-4.0], [4.0], [4.0]]]), "pred_boxes": torch.tensor([[[0.9, 0.9, 0.05, 0.05], [0.8, 0.8, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]]]), "image_logits": {"cancer": torch.tensor([[3.0, -3.0]])}}
    lg, lb = crit(good, targets), crit(bad, targets)
    assert lg["loss_total"] < lb["loss_total"]
    assert lg["loss_giou"] < 1e-4 and lg["loss_bbox"] < 1e-4
    assert lg["loss_img_cancer"] < lb["loss_img_cancer"]
