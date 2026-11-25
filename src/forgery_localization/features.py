"""Model inputs for the RGB, RGB + Noise and RGB + DCT-frequency variants.

These functions reproduce the ``controlled_multiseed_v1`` Colab protocol
exactly, so checkpoints trained in the notebook can be evaluated here:

* RGB: ImageNet mean/std normalization of the resized image in [0, 1].
* Noise: (image - Gaussian5x5(sigma=1, reflect padding)(image)) / ImageNet std.
* Frequency: three log energy maps of an orthonormal 8x8 block DCT on the
  BT.601 grayscale image (bands u+v in [1,3], [4,7], [8,14]), scaled by
  log1p(.)/log(65) and upsampled with nearest neighbour.

The extra features are always computed from the *unnormalized* image after
resizing and after any paired flip augmentation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

MODES = ("rgb", "noise", "frequency")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DCT_BANDS = ((1, 3), (4, 7), (8, 14))


def input_channels(mode: str) -> int:
    """Number of model input channels for a feature mode."""
    _check_mode(mode)
    return 3 if mode == "rgb" else 6


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def _dct_matrix(n: int = 8) -> Tensor:
    k = torch.arange(n).float().view(n, 1)
    x = torch.arange(n).float().view(1, n)
    matrix = torch.cos(math.pi * (x + 0.5) * k / n) * math.sqrt(2 / n)
    matrix[0] /= math.sqrt(2)
    return matrix


def _gaussian_kernel() -> Tensor:
    coordinate = torch.arange(5).float() - 2
    kernel1d = torch.exp(-coordinate.square() / 2)
    kernel1d /= kernel1d.sum()
    return (kernel1d[:, None] * kernel1d[None, :])[None, None].repeat(3, 1, 1, 1)


class FeatureBuilder:
    """Convert a (3, H, W) image in [0, 1] into the model input for one mode."""

    def __init__(self, mode: str) -> None:
        _check_mode(mode)
        self.mode = mode
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        self.gaussian_kernel = _gaussian_kernel()
        self.gray_weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        self.dct = _dct_matrix(8)
        uv = torch.arange(8).view(8, 1) + torch.arange(8).view(1, 8)
        self.band_masks = [(uv >= low) & (uv <= high) for low, high in DCT_BANDS]

    @property
    def channels(self) -> int:
        return input_channels(self.mode)

    def noise(self, image: Tensor) -> Tensor:
        padded = F.pad(image.unsqueeze(0), [2, 2, 2, 2], mode="reflect")
        smooth = F.conv2d(padded, self.gaussian_kernel.to(image), groups=3).squeeze(0)
        return (image - smooth) / self.std.to(image)

    def frequency(self, image: Tensor) -> Tensor:
        gray = (image * self.gray_weights.to(image)).sum(dim=0)
        height, width = gray.shape
        if height % 8 or width % 8:
            raise ValueError("Frequency features require height and width divisible by 8")
        dct = self.dct.to(image)
        blocks = gray.unfold(0, 8, 8).unfold(1, 8, 8)
        energy = (dct @ blocks @ dct.T).square()
        maps = torch.stack([energy[..., band].mean(dim=-1) for band in self.band_masks])
        maps = torch.log1p(maps) / math.log(65)
        return F.interpolate(maps.unsqueeze(0), size=(height, width), mode="nearest").squeeze(0)

    def __call__(self, image: Tensor) -> Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Expected an RGB image tensor with shape (3, H, W)")
        image = image.float()
        rgb = (image - self.mean.to(image)) / self.std.to(image)
        if self.mode == "rgb":
            return rgb.contiguous()
        extra = self.noise(image) if self.mode == "noise" else self.frequency(image)
        return torch.cat([rgb, extra], dim=0).contiguous()
