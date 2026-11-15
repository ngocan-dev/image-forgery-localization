"""Small training/validation loops and validation-only early stopping."""

from dataclasses import dataclass
import math
import torch
from forgery_localization.evaluation.metrics import PixelMetrics


def resolve_device(device="auto"):
    """Choose CUDA when available, otherwise CPU; reject unavailable CUDA."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(device)
    if result.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use device: cpu")
    return result


def _run_epoch(model, loader, criterion, device, optimizer=None, threshold=0.5):
    training = optimizer is not None
    model.train(training)
    metric = PixelMetrics(threshold)
    total_loss, samples = 0.0, 0
    # Validation builds no computation graph and never calls optimizer.step().
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise FloatingPointError("Expected a finite scalar loss")
            if training:
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(f"Nonfinite gradient: {name}")
                optimizer.step()
            total_loss += loss.item() * images.shape[0]
            samples += images.shape[0]
            metric.update(logits.detach(), targets)
    if samples == 0:
        raise ValueError("Cannot run an epoch with an empty data loader")
    return {"loss": total_loss / samples, **metric.compute()}


def train_one_epoch(model, loader, optimizer, criterion, device="cpu", threshold=0.5):
    """Update parameters for one epoch; model must already be on device."""
    return _run_epoch(model, loader, criterion, device, optimizer, threshold)


def validate_one_epoch(model, loader, criterion, device="cpu", threshold=0.5):
    """Compute loss and metrics with eval mode and gradients disabled."""
    return _run_epoch(model, loader, criterion, device, threshold=threshold)


@dataclass
class EarlyStopping:
    """Stop after patience consecutive epochs without a meaningful improvement."""

    patience: int = 10
    mode: str = "max"
    min_delta: float = 0.0
    best: float | None = None
    bad_epochs: int = 0

    def __post_init__(self):
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("patience must be a positive integer")
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be min or max")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")

    def update(self, value):
        if not math.isfinite(value):
            raise ValueError("Early stopping value must be finite")
        improvement = (self.best is None or
                       (value > self.best + self.min_delta if self.mode == "max"
                        else value < self.best - self.min_delta))
        if improvement:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
