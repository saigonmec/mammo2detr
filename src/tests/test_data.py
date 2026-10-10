import numpy as np
import torch

from src.config import Config
from src.data import MammoDetectionDataset, build_dataloaders, collate_fn, load_metadata
from src.data.transforms import build_eval_transform, build_train_transform, final_resize
from src.utils.box_ops import box_cxcywh_to_xyxy


def test_metadata_and_val_split(synthetic_root):
    cfg = Config.load(None, [f"data.root={synthetic_root}", "data.val_fraction=0.25"])
    b = load_metadata(cfg.data, cfg.seed)
    assert set(b.splits) == {"train", "val", "test"}
    train_p = {r.patient_id for r in b["train"]}
    val_p = {r.patient_id for r in b["val"]}
    assert not (train_p & val_p), "patient leakage between train and val"
    assert b.image_class_names["cancer"] == ["0", "1"]
    assert sum(b.image_class_counts["cancer"]) == len(b["train"])
    pos = [r for r in b["train"] if r.image_labels["cancer"] == 1]
    assert all(r.num_boxes > 0 for r in pos)


def test_dataset_item_and_box_roundtrip(synthetic_root):
    cfg = Config.load(None, [f"data.root={synthetic_root}", "data.img_size=[64,48]", "data.resize_mode=letterbox"])
    b = load_metadata(cfg.data, cfg.seed)
    recs = [r for r in b["train"] if r.num_boxes > 0]
    ds = MammoDetectionDataset(recs, cfg.data, build_eval_transform(cfg.data))
    img, mask, t = ds[0]
    assert img.shape == (3, 64, 48) and mask.shape == (64, 48)
    assert mask.any(), "letterbox should pad (images are 96x80)"
    vh, vw = t["size"].tolist()
    oh, ow = t["orig_size"].tolist()
    back = box_cxcywh_to_xyxy(t["boxes"]) * torch.tensor([ow, oh, ow, oh])
    assert np.allclose(back.numpy(), recs[0].boxes_xyxy, atol=1.5)
    ds_tr = MammoDetectionDataset(recs, cfg.data, build_train_transform(cfg.data))
    img2, mask2, t2 = ds_tr[0]
    assert img2.shape == (3, 64, 48) and t2["boxes"].shape[1] == 4


def test_loaders_and_collate(synthetic_root):
    cfg = Config.load(None, [f"data.root={synthetic_root}", "data.img_size=[64,48]", "data.batch_size=4", "data.num_workers=0"])
    bundle, loaders = build_dataloaders(cfg)
    images, masks, targets = next(iter(loaders["train"]))
    assert images.shape == (4, 3, 64, 48) and masks.shape == (4, 64, 48) and len(targets) == 4
    assert all(t["boxes"].shape[1] == 4 for t in targets)


def test_final_resize_stretch():
    cfg = Config.load(None, ["data.img_size=[100,200]", "data.resize_mode=stretch"])
    img = np.zeros((50, 50, 3), np.uint8)
    out, boxes, mask, valid, (sx, sy) = final_resize(img, np.array([[10, 10, 20, 30]], np.float32), cfg.data)
    assert out.shape == (100, 200, 3) and not mask.any() and valid == (100, 200)
    assert np.allclose(boxes, [[40, 20, 80, 60]]) and (sx, sy) == (4.0, 2.0)
