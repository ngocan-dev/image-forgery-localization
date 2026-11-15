"""Stable binary segmentation objectives operating on raw model logits."""

import math
import torch
from torch import nn


def validate_segmentation_tensors(logits, targets):
    """Reject broadcasting, nonbinary targets and nonfinite input values."""
    if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Expected matching (B, 1, H, W) logits and targets")
    if logits.numel() == 0:
        raise ValueError("Empty tensors are not supported")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("Logits and targets must be finite")
    if not ((targets == 0) | (targets == 1)).all():
        raise ValueError("Targets must be binary 0/1")


class BCEWithLogitsLoss(nn.Module):
    """Mean binary cross entropy with stable, internal sigmoid computation."""

    def forward(self, logits, targets):
        validate_segmentation_tensors(logits, targets)
        return nn.functional.binary_cross_entropy_with_logits(logits.float(), targets.float())


class SoftDiceLoss(nn.Module):
    """Mean per-image soft Dice loss, including authentic (empty) targets.

    Smoothing in numerator and denominator makes an empty prediction/target
    pair score one. False foreground on an empty target is penalized. BCE
    supplies useful gradients when Dice alone saturates on empty targets.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("smooth must be finite and positive")
        self.smooth = smooth

    def forward(self, logits, targets):
        validate_segmentation_tensors(logits, targets)
        probabilities = logits.float().sigmoid().flatten(1)
        targets = targets.float().flatten(1)
        intersection = (probabilities * targets).sum(1)
        denominator = probabilities.sum(1) + targets.sum(1)
        return (1 - (2 * intersection + self.smooth) / (denominator + self.smooth)).mean()


class CombinedBCEDiceLoss(nn.Module):
    """Weighted sum of mean BCE and mean per-image soft Dice loss."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        if any(not math.isfinite(w) or w < 0 for w in (bce_weight, dice_weight)):
            raise ValueError("Loss weights must be finite and nonnegative")
        if bce_weight + dice_weight <= 0:
            raise ValueError("At least one loss weight must be positive")
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.bce = BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth)

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)
