"""Image-level (Real/Fake) and per-image pixel metrics in pure NumPy.

Positive class = forged/tampered. Thresholds must be chosen on validation
data and then frozen for test and cross-dataset evaluation.
"""

from __future__ import annotations

import numpy as np


def _arrays(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays of equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return scores, labels


def roc_auc(scores, labels) -> float:
    """Area under the ROC curve (Mann-Whitney U, ties counted as 0.5)."""
    scores, labels = _arrays(scores, labels)
    positives, negatives = scores[labels], scores[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = ranks[labels].sum()
    u = rank_sum - len(positives) * (len(positives) + 1) / 2
    return float(u / (len(positives) * len(negatives)))


def classification_metrics(scores, labels, threshold: float) -> dict:
    """Confusion matrix and Accuracy/Precision/Recall/F1/Specificity/AUC."""
    scores, labels = _arrays(scores, labels)
    predicted = scores >= threshold
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    tn = int((~predicted & ~labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "n_images": int(len(labels)),
        "n_forged": int(labels.sum()),
        "n_authentic": int((~labels).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "auc": roc_auc(scores, labels),
    }


def choose_threshold(scores, labels, criterion: str = "f1") -> float:
    """Threshold maximizing ``f1`` or ``balanced_accuracy`` (ties -> higher threshold)."""
    scores, labels = _arrays(scores, labels)
    if criterion not in ("f1", "balanced_accuracy"):
        raise ValueError("criterion must be f1 or balanced_accuracy")
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best_threshold, best_value = 0.5, -1.0
    for threshold in candidates:
        value = classification_metrics(scores, labels, threshold)[criterion]
        if value > best_value or (value == best_value and threshold > best_threshold):
            best_threshold, best_value = float(threshold), value
    return best_threshold


def pixel_summary(rows: list[dict]) -> dict:
    """Aggregate per-image pixel counts (keys: tp, fp, fn, n_pixels, forged).

    * micro Dice / IoU: pooled TP/FP/FN over all images;
    * mean tampered Dice / IoU: per-image average over forged images only;
    * authentic FP pixel %: false-positive pixels / pixels on authentic images;
    * authentic images with any FP %: share of authentic images with >=1 FP pixel.
    """
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    tampered = [row for row in rows if row["forged"]]
    authentic = [row for row in rows if not row["forged"]]

    def dice(row):
        return 2 * row["tp"] / max(2 * row["tp"] + row["fp"] + row["fn"], 1)

    def iou(row):
        return row["tp"] / max(row["tp"] + row["fp"] + row["fn"], 1)

    auth_pixels = sum(row["n_pixels"] for row in authentic)
    return {
        "n_images": len(rows),
        "n_tampered": len(tampered),
        "n_authentic": len(authentic),
        "micro_dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "micro_iou": tp / max(tp + fp + fn, 1),
        "pixel_precision": tp / max(tp + fp, 1),
        "pixel_recall": tp / max(tp + fn, 1),
        "mean_tampered_dice": float(np.mean([dice(r) for r in tampered])) if tampered else float("nan"),
        "mean_tampered_iou": float(np.mean([iou(r) for r in tampered])) if tampered else float("nan"),
        "authentic_fp_pixel_pct": (100 * sum(r["fp"] for r in authentic) / auth_pixels
                                   if auth_pixels else float("nan")),
        "authentic_any_fp_pct": (100 * np.mean([r["fp"] > 0 for r in authentic])
                                 if authentic else float("nan")),
    }
