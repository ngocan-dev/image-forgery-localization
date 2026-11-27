"""Single-image inference shared by scripts/predict.py, scripts/evaluate.py and the demo.

Pipeline (identical preprocessing to training):

    image -> RGB, resize 256x256 bilinear, /255 -> FeatureBuilder(mode)
          -> model -> logits -> sigmoid (256x256 probability map)
          -> image score at model resolution (or the image-level head's
             probability, score method ``cls``) -> Real/Fake
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
from forgery_localization.models.factory import load_checkpoint, model_label, split_outputs

CLS_SCORE = "cls"   # score method: probability of the image-level classification head

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
        if self.thresholds["score_method"] == CLS_SCORE and not self.has_cls_head:
            raise ValueError("Score method 'cls' needs a checkpoint trained with --cls-head")

    @property
    def name(self) -> str:
        return f"{model_label(self.config)} | {self.config['mode']} | seed {self.config.get('seed')}"

    @property
    def has_cls_head(self) -> bool:
        return bool(self.config.get("cls_head", False))

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(B, 3, H, W) images in [0, 1] -> ((B, 1, H, W) pixel probabilities, (B,) head probabilities or None)."""
        inputs = torch.stack([self.features(image) for image in images.cpu()])
        logits, cls_logits = split_outputs(self.model(inputs.to(self.device)))
        cls = None if cls_logits is None else cls_logits.float().sigmoid()[:, 0].cpu()
        return logits.float().sigmoid().cpu(), cls

    def probabilities_from_tensors(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) images in [0, 1] -> (B, 1, H, W) probabilities."""
        return self.forward(images)[0]

    def predict(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, PredictionSummary]:
        """Return (probability map, binary mask, summary) at the ORIGINAL image size."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = image_to_tensor(image, self.size)
        probabilities, cls = self.forward(tensor.unsqueeze(0))
        t = self.thresholds
        if t["score_method"] == CLS_SCORE:
            score = float(cls[0])
        else:
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
        f"**Result: {summary.label}** (decision margin {100 * summary.confidence:.1f}%: "
        f"0% = at the threshold, 100% = maximally far from it)",
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
    lines.append("- Note: the score and margin are not calibrated probabilities; an empty mask does not prove authenticity.")
    return "\n".join(lines)
