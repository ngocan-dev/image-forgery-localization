"""Lightweight U-Net for pixel-level image forgery localization."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two padded convolutions learn features without changing image size."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    """Upsample coarse features and combine them with an encoder skip."""

    def __init__(self, in_channels: int, skip_channels: int) -> None:
        super().__init__()
        # Reduce channels before combining with the high-resolution skip.
        self.reduce = nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        self.conv = DoubleConv(2 * skip_channels, skip_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.reduce(x)
        # Match the skip exactly, including for odd input heights/widths.
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((skip, x), dim=1)  # Concatenate along the channel axis.
        return self.conv(x)


class UNet(nn.Module):
    """Four-level U-Net returning raw, full-resolution segmentation logits.

    Defaults map (B, 3, H, W) RGB images to (B, 1, H, W) binary logits.
    Height and width must each be at least 16 for four pooling operations.
    A base width of 16 keeps the model small; channels grow to 256 at the
    bottleneck. No pretrained weights or device selection are implicit.
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16
    ) -> None:
        super().__init__()
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels),
                            ("base_channels", base_channels)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.in_channels = in_channels
        c = base_channels
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Encoder: halve spatial dimensions and increase channels at each level.
        self.encoder1 = DoubleConv(in_channels, c)
        self.encoder2 = DoubleConv(c, 2 * c)
        self.encoder3 = DoubleConv(2 * c, 4 * c)
        self.encoder4 = DoubleConv(4 * c, 8 * c)
        self.bottleneck = DoubleConv(8 * c, 16 * c)
        # Decoder: restore resolution using the corresponding encoder features.
        self.decoder4 = UpBlock(16 * c, 8 * c)
        self.decoder3 = UpBlock(8 * c, 4 * c)
        self.decoder2 = UpBlock(4 * c, 2 * c)
        self.decoder1 = UpBlock(2 * c, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict logits; apply sigmoid separately only when probabilities are needed."""
        if x.ndim != 4:
            raise ValueError("Expected input shape (B, C, H, W)")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        if min(x.shape[-2:]) < 16:
            raise ValueError("Input height and width must each be at least 16")
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        skip4 = self.encoder4(self.pool(skip3))
        x = self.bottleneck(self.pool(skip4))
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        # BCEWithLogitsLoss includes sigmoid internally, so do not apply it here.
        return self.head(x)
