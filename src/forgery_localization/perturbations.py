"""Deterministic image perturbations for robustness tests (pure PIL/NumPy).

A specification is ``kind:value``:

* ``jpeg:Q``      re-encode as JPEG with quality Q (1-100);
* ``resize:F``    downscale by factor F (0 < F < 1), then upscale back to the
                  original size, so ground-truth masks stay aligned;
* ``noise:S``     add Gaussian noise with standard deviation S (0-255 scale);
* ``blur:R``      Gaussian blur with radius R;
* ``none``        unchanged image.

Perturbations are applied to the full-resolution image before the model's
resize. Noise is seeded by the image key, so every model sees the same noise.
"""

from __future__ import annotations

import hashlib
import io
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

Perturbation = Callable[[Image.Image, str], Image.Image]


def parse_perturbation(spec: str) -> Perturbation | None:
    """Return a ``(image, key) -> image`` callable, or None for ``none``."""
    spec = spec.strip().lower()
    if spec in ("", "none"):
        return None
    kind, _, raw = spec.partition(":")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"Perturbation must look like kind:value, got {spec!r}") from error

    if kind == "jpeg":
        if not 1 <= value <= 100:
            raise ValueError("JPEG quality must be within [1, 100]")
        quality = int(value)

        def apply(image, key):
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB")
        return apply
    if kind == "resize":
        if not 0 < value < 1:
            raise ValueError("Resize factor must be within (0, 1)")

        def apply(image, key):
            width, height = image.size
            small = image.resize((max(1, round(width * value)), max(1, round(height * value))),
                                 Image.Resampling.BILINEAR)
            return small.resize((width, height), Image.Resampling.BILINEAR)
        return apply
    if kind == "noise":
        if value < 0:
            raise ValueError("Noise standard deviation must be non-negative")

        def apply(image, key):
            seed = int.from_bytes(hashlib.sha256(f"noise|{value}|{key}".encode()).digest()[:8], "little")
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            noisy = pixels + np.random.default_rng(seed).normal(0.0, value, pixels.shape)
            return Image.fromarray(np.clip(noisy, 0, 255).round().astype(np.uint8))
        return apply
    if kind == "blur":
        if value < 0:
            raise ValueError("Blur radius must be non-negative")

        def apply(image, key):
            return image.convert("RGB").filter(ImageFilter.GaussianBlur(value))
        return apply
    raise ValueError(f"Unknown perturbation kind {kind!r}; use jpeg, resize, noise, blur or none")


def perturbation_tag(spec: str) -> str:
    """File-system friendly name, e.g. ``jpeg:70`` -> ``jpeg70``, ``resize:0.5`` -> ``resize0p5``."""
    return spec.strip().lower().replace(":", "").replace(".", "p") or "none"


class RandomDegradation:
    """Training-time robustness augmentation: a random post-processing chain per image.

    For every (seed, epoch, image key) the same choice is made, so runs are
    reproducible and resumable. With probability ``probability`` the image gets
    one or two of: JPEG re-compression (quality 50-95), down-and-up scaling
    (factor 0.5-0.9), Gaussian noise (sigma 1-8) or Gaussian blur (radius
    0.3-1.5); otherwise it is unchanged. The image size never changes, so the
    ground-truth mask stays aligned. ``set_epoch`` must be called before each
    epoch (DataLoader workers are re-created per epoch and copy the new value).
    """

    KINDS = ("jpeg", "resize", "noise", "blur")

    def __init__(self, seed: int, probability: float = 0.5) -> None:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be within [0, 1]")
        self.seed, self.probability, self.epoch = seed, probability, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def specs(self, key: str) -> list[str]:
        """The perturbation specs applied to ``key`` in the current epoch (may be empty)."""
        digest = hashlib.sha256(f"degrade|{self.seed}|{self.epoch}|{key}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        if rng.random() >= self.probability:
            return []
        kinds = rng.choice(self.KINDS, size=1 + int(rng.random() < 0.3), replace=False)
        specs = []
        for kind in kinds:
            if kind == "jpeg":
                specs.append(f"jpeg:{int(rng.integers(50, 96))}")
            elif kind == "resize":
                specs.append(f"resize:{rng.uniform(0.5, 0.9):.3f}")
            elif kind == "noise":
                specs.append(f"noise:{rng.uniform(1.0, 8.0):.2f}")
            else:
                specs.append(f"blur:{rng.uniform(0.3, 1.5):.2f}")
        # JPEG last: social-media style re-encoding happens after other edits.
        return sorted(specs, key=lambda spec: spec.startswith("jpeg"))

    def __call__(self, image: Image.Image, key: str) -> Image.Image:
        for spec in self.specs(key):
            image = parse_perturbation(spec)(image, f"{key}|{self.epoch}")
        return image
