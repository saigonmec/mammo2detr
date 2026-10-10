import torch
import torch.nn as nn

from src.config import DenoisingConfig
from src.models.denoising import build_dn_queries, dn_matched_indices


def _targets():
    return [
        {"boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1]]), "labels": torch.tensor([0, 0])},
        {"boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]), "labels": torch.tensor([0])},
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)},
    ]


def test_mask_semantics_and_shapes():
    enc = nn.Embedding(2, 8)
    dn = build_dn_queries(_targets(), DenoisingConfig(num_groups=2, contrastive=True), 1, enc, num_matching_queries=3, num_extra_queries=1)
    assert dn is not None
    content, boxes_unsig, mask, meta = dn
    P, gs = meta.pad_size, meta.group_size
    assert P == 2 * 2 * 2 and content.shape == (3, P, 8) and boxes_unsig.shape == (3, P, 4)
    assert mask.shape == (P + 4, P + 4)
    assert mask[P:, :P].all() and not mask[P:, P:].any()  # matching/cls never see DN
    assert not mask[:gs, :gs].any() and mask[:gs, gs:P].all() and mask[gs:P, :gs].all()  # groups isolated
    assert not mask[:P, P:].any()  # DN may see matching queries
    idx = dn_matched_indices(meta, _targets())
    assert idx[0][0].tolist() == [0, 1, 4, 5] and idx[0][1].tolist() == [0, 1, 0, 1]
    assert idx[1][0].tolist() == [0, 4]
    assert len(idx[2][0]) == 0
    # positive slots hold noised versions of the GT boxes (close to them), padded slots are zeros
    pos = boxes_unsig[0, idx[0][0]].sigmoid()
    gt = _targets()[0]["boxes"][idx[0][1]]
    assert (pos - gt).abs().max() < 0.3
    assert torch.all(boxes_unsig[2] == 0)


def test_no_boxes_returns_none():
    enc = nn.Embedding(2, 8)
    t = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    assert build_dn_queries(t, DenoisingConfig(), 1, enc, 3) is None


def test_non_contrastive_layout():
    enc = nn.Embedding(2, 8)
    dn = build_dn_queries(_targets(), DenoisingConfig(num_groups=3, contrastive=False), 1, enc, 3)
    _, _, mask, meta = dn
    assert meta.pad_size == 3 * 2 and meta.group_size == 2
    idx = dn_matched_indices(meta, _targets())
    assert idx[0][0].tolist() == [0, 1, 2, 3, 4, 5]
