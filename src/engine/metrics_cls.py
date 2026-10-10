"""Image-level classification metrics (numpy/scipy only)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties handled with average ranks)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_score)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    tp = np.cumsum(y_true[order])
    precision = tp / np.arange(1, len(tp) + 1)
    return float((precision * y_true[order]).sum() / y_true.sum())


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_t, y_s = y_true[order], y_score[order]
    distinct = np.where(np.diff(y_s))[0]
    idx = np.r_[distinct, len(y_s) - 1]
    tps = np.cumsum(y_t)[idx]
    fps = (1 + idx) - tps
    tpr = tps / max(y_t.sum(), 1)
    fpr = fps / max((1 - y_t).sum(), 1)
    return np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[np.inf, y_s[idx]]


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    """Threshold maximising sensitivity + specificity - 1; returns (thr, sens, spec)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(1 - fpr[i])


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)):
        cm[t, p] += 1
    return cm


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, positive_index: int = 1, class_names: list[str] | None = None) -> dict[str, float]:
    """Return a flat metric dict. ``probs``: [N, C] softmax probabilities."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    n, c = probs.shape
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, c)
    out: dict[str, float] = {"n": float(n), "accuracy": float((y_pred == y_true).mean()) if n else float("nan")}
    recalls = np.array([cm[i, i] / cm[i].sum() if cm[i].sum() else np.nan for i in range(c)])
    out["balanced_accuracy"] = float(np.nanmean(recalls))
    pos = positive_index if positive_index < c else c - 1
    tp = cm[pos, pos]
    fn = cm[pos].sum() - tp
    fp = cm[:, pos].sum() - tp
    tn = cm.sum() - tp - fn - fp
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    out["precision"] = float(tp / (tp + fp)) if (tp + fp) else 0.0
    out["f1"] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    if c == 2:
        score = probs[:, pos]
        out["auc"] = binary_auc((y_true == pos).astype(int), score)
        out["ap"] = average_precision((y_true == pos).astype(int), score)
        thr, sens, spec = youden_threshold((y_true == pos).astype(int), score)
        out["youden_threshold"], out["sensitivity_at_youden"], out["specificity_at_youden"] = thr, sens, spec
    else:
        aucs = [binary_auc((y_true == k).astype(int), probs[:, k]) for k in range(c)]
        out["auc"] = float(np.nanmean(aucs))
        for k, a in enumerate(aucs):
            out[f"auc_class{k}"] = float(a)
    for k in range(c):
        name = class_names[k] if class_names and k < len(class_names) else str(k)
        out[f"recall_{name}"] = float(recalls[k])
    out["confusion"] = cm.tolist()  # type: ignore[assignment]
    return out
