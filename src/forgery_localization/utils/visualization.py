"""Display a dataset tensor pair without changing its ground-truth mask."""

import numpy as np
import torch


def visualize_sample(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.4):
    """Return a Matplotlib figure with RGB, ground truth and red overlay panels.

    Accepts one (3,H,W) image in [0,1] and a binary (1,H,W) mask.
    Call plt.show() interactively or fig.savefig(path), then plt.close(fig).
    """
    import matplotlib.pyplot as plt

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape (3, H, W)")
    if mask.shape != (1, *image.shape[1:]):
        raise ValueError("mask must have shape (1, H, W) matching image")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    rgb = image.detach().cpu().permute(1, 2, 0).numpy()
    binary = mask.detach().cpu().numpy()[0]
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("image must contain finite values in [0, 1]")
    if not np.isin(binary, [0, 1]).all():
        raise ValueError("mask must be binary 0/1")
    overlay = rgb.copy()
    foreground = binary.astype(bool)
    overlay[foreground] = (1 - alpha) * rgb[foreground] + alpha * np.array([1, 0, 0])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, data, title in zip(axes, (rgb, binary, overlay),
                                 ("RGB image", "Ground-truth mask", "Manipulation overlay")):
        axis.imshow(data, cmap="gray" if data.ndim == 2 else None, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    return figure
