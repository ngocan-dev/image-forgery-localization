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
