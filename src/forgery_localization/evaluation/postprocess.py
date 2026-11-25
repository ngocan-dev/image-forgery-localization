"""Turn a pixel probability map into the user-facing outputs of topic 2.

From one sigmoid map the system reports:

* a binary manipulation mask (pixel threshold);
* an image-level score and a Real/Fake decision (image threshold, chosen on
  the validation set, never on the test set);
* bounding boxes of connected suspicious regions;
* the suspicious area as a fraction / percentage of the image.

All functions are pure NumPy so they can be tested without a GPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict, field

import numpy as np

SCORE_METHODS = ("max", "top1pct_mean", "area_fraction")


def _check_probability_map(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim == 3 and probabilities.shape[0] == 1:
        probabilities = probabilities[0]
    if probabilities.ndim != 2:
        raise ValueError("Expected a (H, W) probability map")
    if not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return probabilities


def image_score(probabilities: np.ndarray, method: str = "top1pct_mean",
                pixel_threshold: float = 0.5) -> float:
    """Aggregate a pixel map into one image-level manipulation score in [0, 1].

    ``max``           highest pixel probability (sensitive to single noisy pixels);
    ``top1pct_mean``  mean of the 1% most suspicious pixels (default, more stable);
    ``area_fraction`` fraction of pixels >= pixel_threshold.
    """
    probabilities = _check_probability_map(probabilities)
    flat = probabilities.ravel()
    if method == "max":
        return float(flat.max())
    if method == "top1pct_mean":
        k = max(1, int(round(0.01 * flat.size)))
        return float(np.partition(flat, flat.size - k)[-k:].mean())
    if method == "area_fraction":
        return float((flat >= pixel_threshold).mean())
    raise ValueError(f"method must be one of {SCORE_METHODS}")


def connected_regions(mask: np.ndarray, min_area: int = 1) -> list[dict]:
    """8-connected components of a binary mask as boxes, largest first.

    Each region is ``{"x_min", "y_min", "x_max", "y_max", "area_px"}`` with
    inclusive pixel coordinates.
    """
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError("Expected a (H, W) mask")
    height, width = mask.shape
    seen = np.zeros_like(mask)
    regions = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if seen[start_y, start_x]:
            continue
        seen[start_y, start_x] = True
        queue = deque([(start_y, start_x)])
        area, y0, y1, x0, x1 = 0, start_y, start_y, start_x, start_x
        while queue:
            y, x = queue.popleft()
            area += 1
            y0, y1, x0, x1 = min(y0, y), max(y1, y), min(x0, x), max(x1, x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        if area >= min_area:
            regions.append(dict(x_min=int(x0), y_min=int(y0), x_max=int(x1),
                                y_max=int(y1), area_px=int(area)))
    return sorted(regions, key=lambda region: region["area_px"], reverse=True)


@dataclass
class PredictionSummary:
    """Everything topic 2 asks the system to return for one image."""

    label: str                    # "Forged" or "Authentic"
    confidence: float             # probability-like confidence in ``label``
    image_score: float            # manipulation score in [0, 1]
    score_method: str
    image_threshold: float
    pixel_threshold: float
    suspicious_area_fraction: float
    suspicious_area_pct: float
    num_regions: int
    regions: list[dict] = field(default_factory=list)
    height: int = 0
    width: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_prediction(probabilities: np.ndarray, *, pixel_threshold: float = 0.5,
                         image_threshold: float = 0.5, score_method: str = "top1pct_mean",
                         min_region_fraction: float = 0.001,
                         score: float | None = None) -> tuple[np.ndarray, PredictionSummary]:
    """Return (binary mask, summary).

    ``score`` lets callers pass an image score computed at model resolution
    (as during evaluation) while boxes/area use an upsampled map.

    Regions smaller than ``min_region_fraction`` of the image are dropped from
    the box list (they still count in the area fraction). An image is
    reported as Forged when ``image_score >= image_threshold``. The confidence
    is ``score`` for Forged and ``1 - score`` for Authentic; it is a ranking
    score, not a calibrated probability.
    """
    probabilities = _check_probability_map(probabilities)
    for name, value in (("pixel_threshold", pixel_threshold), ("image_threshold", image_threshold)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be within [0, 1]")
    mask = probabilities >= pixel_threshold
    height, width = mask.shape
    if score is None:
        score = image_score(probabilities, score_method, pixel_threshold)
    forged = score >= image_threshold
    min_area = max(1, int(round(min_region_fraction * mask.size)))
    regions = connected_regions(mask, min_area=min_area) if forged else []
    area_fraction = float(mask.mean())
    summary = PredictionSummary(
        label="Forged" if forged else "Authentic",
        confidence=float(score if forged else 1 - score),
        image_score=float(score),
        score_method=score_method,
        image_threshold=float(image_threshold),
        pixel_threshold=float(pixel_threshold),
        suspicious_area_fraction=area_fraction,
        suspicious_area_pct=100 * area_fraction,
        num_regions=len(regions),
        regions=regions,
        height=height,
        width=width,
    )
    return mask, summary
