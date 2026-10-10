import torch
from torchvision.ops import box_iou as tv_iou, generalized_box_iou as tv_giou

from src.utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    box_xywh_to_xyxy,
    box_xyxy_to_cxcywh,
    boxes_center_inside,
    generalized_box_iou,
    generalized_box_iou_pairwise,
    inverse_sigmoid,
)


def _rand_xyxy(n):
    a = torch.rand(n, 4)
    return torch.cat([a[:, :2], a[:, :2] + a[:, 2:] * 0.5 + 0.01], 1)


def test_iou_and_giou_match_torchvision():
    a, b = _rand_xyxy(7), _rand_xyxy(5)
    assert torch.allclose(box_iou(a, b)[0], tv_iou(a, b), atol=1e-6)
    assert torch.allclose(generalized_box_iou(a, b), tv_giou(a, b), atol=1e-6)
    assert torch.allclose(generalized_box_iou_pairwise(a[:5], b), tv_giou(a[:5], b).diagonal(), atol=1e-6)


def test_conversions_roundtrip():
    a = _rand_xyxy(6)
    assert torch.allclose(box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(a)), a, atol=1e-6)
    xywh = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(box_xywh_to_xyxy(xywh), torch.tensor([[1.0, 2.0, 4.0, 6.0]]))


def test_inverse_sigmoid():
    x = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    y = torch.sigmoid(inverse_sigmoid(x))
    assert torch.allclose(y[1:4], x[1:4], atol=1e-4)
    assert torch.isfinite(y).all()


def test_center_inside():
    a = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.8, 0.8, 0.9, 0.9]])
    b = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    m = boxes_center_inside(a, b)
    assert m.tolist() == [[True], [False]]
