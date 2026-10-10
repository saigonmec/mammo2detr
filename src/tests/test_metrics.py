import numpy as np

from src.engine.metrics_cls import average_precision, binary_auc, classification_metrics
from src.engine.metrics_det import average_precision_det, froc, localization_recall


def test_auc_known_values():
    y = np.array([0, 0, 1, 1])
    assert binary_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert binary_auc(y, np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert abs(average_precision(y, np.array([0.1, 0.4, 0.35, 0.8])) - 0.8333) < 1e-3


def test_classification_metrics():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    m = classification_metrics(np.array([0, 1, 1, 0]), probs)
    assert m["accuracy"] == 0.5 and m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert 0.0 <= m["auc"] <= 1.0


def _images():
    return [
        {"pred_boxes": np.array([[10, 10, 50, 50], [100, 100, 120, 120]]), "pred_scores": np.array([0.9, 0.4]), "pred_labels": np.array([0, 0]), "gt_boxes": np.array([[12, 12, 48, 48]]), "gt_labels": np.array([0])},
        {"pred_boxes": np.array([[0, 0, 10, 10]]), "pred_scores": np.array([0.3]), "pred_labels": np.array([0]), "gt_boxes": np.zeros((0, 4)), "gt_labels": np.zeros(0)},
        {"pred_boxes": np.zeros((0, 4)), "pred_scores": np.zeros(0), "pred_labels": np.zeros(0), "gt_boxes": np.array([[200, 200, 240, 240]]), "gt_labels": np.array([0])},
    ]


def test_average_precision_det():
    ap = average_precision_det(_images(), 0.5)
    assert ap["n_gt"] == 2 and abs(ap["ap"] - 0.5) < 0.02  # one GT found at full precision, one missed
    perfect = [{"pred_boxes": im["gt_boxes"], "pred_scores": np.ones(len(im["gt_boxes"])), "pred_labels": np.zeros(len(im["gt_boxes"])), "gt_boxes": im["gt_boxes"], "gt_labels": im["gt_labels"]} for im in _images()]
    assert average_precision_det(perfect, 0.5)["ap"] == 1.0


def test_froc_curve():
    fr = froc(_images(), [0.25, 0.5, 1.0, 2.0], "iou", 0.25)
    # at threshold 0.9: 1/2 GT found, 0 FP -> fppi 0 ; at 0.4: 1/2 found, 1 FP over 3 images (0.33) ; at 0.3: 2 FP (0.67)
    assert fr["sens@fppi0.25"] == 0.5 and fr["sens@fppi1"] == 0.5
    assert abs(fr["froc_mean"] - 0.5) < 1e-9
    assert fr["n_gt"] == 2 and fr["n_images"] == 3


def test_localization_recall():
    r = localization_recall(_images(), 1, "iou", 0.25)
    assert r["img_recall@1"] == 0.5 and r["lesion_recall@1"] == 0.5
