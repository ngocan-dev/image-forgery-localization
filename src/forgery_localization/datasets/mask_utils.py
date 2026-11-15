"""Shared mask decoding: channel validation precedes any approved thresholding."""
from pathlib import Path
import numpy as np
from PIL import Image


def read_mask_values(path: str | Path) -> np.ndarray:
    """Return unmodified grayscale values; never infer polarity or discard alpha."""
    with Image.open(path) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f'Non-default mask orientation: {path}')
        if image.mode == 'P':
            image = image.convert('RGBA' if 'transparency' in image.info else 'RGB')
        if image.mode not in ('1', 'L', 'RGB', 'RGBA'):
            raise ValueError(f'Unsupported mask mode {image.mode}: {path}')
        # PNG transparency can also be stored separately from pixel channels.
        if 'transparency' in image.info and image.mode in ('L', 'RGB'):
            image = image.convert('RGBA')
        values = np.array(image)
    if values.ndim == 3:
        if values.shape[2] == 4 and not np.all(values[..., 3] == 255):
            raise ValueError(f'Mask alpha must be 255 everywhere: {path}')
        rgb = values[..., :3]
        if not np.all(rgb == rgb[..., :1]):
            raise ValueError(f'Mask RGB channels must be identical: {path}')
        values = rgb[..., 0]
    return values


def binarize_mask(values: np.ndarray, *, threshold_approved=False, threshold=127) -> np.ndarray:
    """Preserve binary 0/1 and 0/255; otherwise require approval for value > threshold.

    Thresholding does not determine whether white or black is manipulated.
    Values exactly equal to the threshold become zero.
    """
    if type(threshold_approved) is not bool:
        raise ValueError('threshold_approved must be a boolean')
    if type(threshold) is not int or not 0 <= threshold <= 254:
        raise ValueError('mask_threshold must be an integer between 0 and 254')
    unique = set(np.unique(values).tolist())
    if unique <= {0, 1} or unique <= {0, 255}:
        return values > 0
    if not threshold_approved:
        raise ValueError('Mask is not binary 0/1 or 0/255; intermediate values require explicit threshold approval')
    return values > threshold


def threshold_options(row):
    """Parse optional manifest/review settings without treating 'false' as true."""
    approval = str(row.get('threshold_approved', '')).strip().lower()
    if approval not in ('', 'true', 'false'):
        raise ValueError('threshold_approved must be true or false')
    raw = str(row.get('mask_threshold', '')).strip()
    try:
        threshold = int(raw) if raw else 127
    except ValueError as error:
        raise ValueError('mask_threshold must be an integer') from error
    if not 0 <= threshold <= 254:
        raise ValueError('mask_threshold must be between 0 and 254')
    return dict(threshold_approved=approval == 'true', threshold=threshold)
