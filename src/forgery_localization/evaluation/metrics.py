"""Dataset-wide (micro) binary pixel metrics accumulated from counts."""

import torch
from forgery_localization.training.losses import validate_segmentation_tensors


class PixelMetrics:
    """Accumulate TP/FP/FN, avoiding averages of per-batch ratios.

    Both-empty masks score 1 for all metrics. Otherwise an undefined precision
    or recall scores 0. Dice and F1 coincide for binary pixel segmentation.
    """

    def __init__(self, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be strictly between 0 and 1")
        self.threshold = threshold
        self.tp = self.fp = self.fn = 0

    @torch.no_grad()
    def update(self, logits, targets):
        validate_segmentation_tensors(logits, targets)
        prediction = logits.sigmoid() >= self.threshold
        truth = targets.bool()
        self.tp += int((prediction & truth).sum().item())
        self.fp += int((prediction & ~truth).sum().item())
        self.fn += int((~prediction & truth).sum().item())

    def compute(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        if tp + fp + fn == 0:
            return dict(precision=1.0, recall=1.0, f1=1.0, dice=1.0, iou=1.0)
        return {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn),
            "dice": 2 * tp / (2 * tp + fp + fn),
            "iou": tp / (tp + fp + fn),
        }


def pixel_metrics(logits, targets, threshold=0.5):
    """Convenience function for one tensor pair; epoch loops use PixelMetrics."""
    accumulator = PixelMetrics(threshold)
    accumulator.update(logits, targets)
    return accumulator.compute()
