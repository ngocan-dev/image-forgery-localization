"""Single-image inference shared by scripts/predict.py, scripts/evaluate.py and the demo.

Pipeline (identical preprocessing to training):

    image -> RGB, resize 256x256 bilinear, /255 -> FeatureBuilder(mode)
          -> model -> logits -> sigmoid (256x256 probability map)
          -> image score at model resolution -> Real/Fake
          -> probability map resized bilinearly to the original size
          -> threshold -> mask, connected regions (boxes), area fraction
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch.nn import functional as F

from forgery_localization.evaluation.postprocess import (
    PredictionSummary, image_score, summarize_prediction,
)
from forgery_localization.features import FeatureBuilder
from forgery_localization.models.factory import load_checkpoint

DEFAULT_THRESHOLDS = {
    "pixel_threshold": 0.5,
    "image_threshold": 0.5,
    "score_method": "top1pct_mean",
    "calibrated_on": None,
}


def load_thresholds(path: str | Path | None) -> dict:
    """Read thresholds.json written by ``scripts/evaluate.py --calibrate``."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    if path is not None:
        thresholds.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return thresholds


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def image_to_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """RGB PIL image -> float tensor (3, H, W) in [0, 1], bilinear like CASIADataset."""
    height, width = size
    array = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
                       dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class Predictor:
    """Load a checkpoint once; predict masks and reports for many images."""

    def __init__(self, checkpoint: str | Path, thresholds: dict | None = None,
                 device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.model, self.config = load_checkpoint(checkpoint, self.device)
        self.features = FeatureBuilder(self.config["mode"])
        self.size = tuple(self.config.get("image_size", (256, 256)))
        self.thresholds = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))

    @property
    def name(self) -> str:
        return f"{self.config['architecture']} | {self.config['mode']} | seed {self.config.get('seed')}"

    @torch.inference_mode()
    def probabilities_from_tensors(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) images in [0, 1] -> (B, 1, H, W) probabilities."""
        inputs = torch.stack([self.features(image) for image in images.cpu()])
        logits = self.model(inputs.to(self.device))
        return logits.float().sigmoid().cpu()

    def predict(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, PredictionSummary]:
        """Return (probability map, binary mask, summary) at the ORIGINAL image size."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = image_to_tensor(image, self.size)
        probabilities = self.probabilities_from_tensors(tensor.unsqueeze(0))
        t = self.thresholds
        score = image_score(probabilities[0, 0].numpy(), t["score_method"], t["pixel_threshold"])
        full = F.interpolate(probabilities, size=(image.height, image.width), mode="bilinear",
                             align_corners=False)[0, 0].clamp(0, 1).numpy()
        mask, summary = summarize_prediction(
            full, pixel_threshold=t["pixel_threshold"], image_threshold=t["image_threshold"],
            score_method=t["score_method"], score=score,
        )
        return full, mask, summary


def render_overlay(image: Image.Image, mask: np.ndarray, summary: PredictionSummary | None = None,
                   alpha: float = 0.45) -> Image.Image:
    """Red translucent mask plus yellow bounding boxes on top of the image."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("mask must match the image size")
    blended = rgb.copy()
    blended[mask] = (1 - alpha) * rgb[mask] + alpha * np.array([255, 0, 0], dtype=np.float32)
    result = Image.fromarray(blended.clip(0, 255).astype(np.uint8))
    if summary is not None:
        draw = ImageDraw.Draw(result)
        width = max(2, round(min(result.size) / 200))
        for index, region in enumerate(summary.regions, start=1):
            box = (region["x_min"], region["y_min"], region["x_max"], region["y_max"])
            draw.rectangle(box, outline=(255, 220, 0), width=width)
            draw.text((region["x_min"] + 3, region["y_min"] + 2), f"#{index}", fill=(255, 220, 0))
    return result


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)


def report_text(summary: PredictionSummary, model_name: str = "") -> str:
    """Short human-readable report (Markdown)."""
    lines = [
        f"**Result: {summary.label}** (confidence {100 * summary.confidence:.1f}%)",
        f"- Image manipulation score: {summary.image_score:.3f} "
        f"(method `{summary.score_method}`, threshold {summary.image_threshold:.3f})",
        f"- Suspicious area: {summary.suspicious_area_pct:.2f}% of the image",
        f"- Suspicious regions (boxes): {summary.num_regions}",
    ]
    for index, region in enumerate(summary.regions[:10], start=1):
        lines.append(
            f"  - #{index}: x {region['x_min']}-{region['x_max']}, y {region['y_min']}-{region['y_max']}, "
            f"{100 * region['area_px'] / max(summary.height * summary.width, 1):.2f}% area"
        )
    if model_name:
        lines.append(f"- Model: {model_name}")
    lines.append("- Note: scores are not calibrated probabilities; an empty mask does not prove authenticity.")
    return "\n".join(lines)
