"""Explicit, validated CASIA image-mask manifests and source-group splits.

No filename convention is assumed. Verification fields record the curator's
assertion; software cannot establish semantic mask correctness or provenance.
"""

import csv
import hashlib
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from forgery_localization.datasets.mask_utils import read_mask_values, binarize_mask, threshold_options


_FIELDS = {"image_path", "mask_path", "label", "source_group", "mask_verified", "mask_source"}
_LABELS = {"au": "authentic", "authentic": "authentic", "tp": "tampered", "tampered": "tampered"}


def _read_rgb(path: Path) -> Image.Image:
    """Decode in stored pixel orientation; never rotate images independently of masks."""
    with Image.open(path) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f"Non-default EXIF orientation requires paired preprocessing: {path}")
        return image.convert("RGB")


def _read_mask(path: Path, size: tuple[int, int], **options) -> Image.Image:
    values = read_mask_values(path)
    if values.shape != (size[1], size[0]):
        raise ValueError(f"Mask dimensions mismatch image {size}: {path}")
    binary = binarize_mask(values, **options)
    if not binary.any():
        raise ValueError(f"Tampered mask has no foreground: {path}")
    return Image.fromarray(binary.astype(np.uint8))


class CASIADataset(Dataset):
    """Return (RGB image, binary mask) tensors from an explicit CSV manifest.

    Relative image/mask paths resolve against ``root`` (default: current working
    directory), NOT the manifest directory. Paths may also be absolute. Each
    tampered row must declare mask_verified=true and a nonempty mask_source.
    All files are decoded and validated eagerly, then read again on access.
    """

    def __init__(self, manifest: str | Path, root: str | Path = ".",
                 image_size: tuple[int, int] = (256, 256), perturbation=None) -> None:
        if len(image_size) != 2 or any(type(n) is not int or n < 1 for n in image_size):
            raise ValueError("image_size must contain positive integer (height, width)")
        self.image_size = tuple(image_size)
        # Optional (PIL image, key) -> PIL image, applied before resizing (robustness tests)
        self.perturbation = perturbation
        root = Path(root).resolve()
        manifest = Path(manifest)
        if not manifest.is_absolute():
            manifest = root / manifest
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not _FIELDS.issubset(reader.fieldnames or []):
                raise ValueError(f"Manifest requires columns: {', '.join(sorted(_FIELDS))}")
            rows = list(reader)
        if not rows:
            raise ValueError("Manifest contains no samples")
        records, seen_images, seen_masks, content_groups = [], set(), set(), {}
        for line, row in enumerate(rows, start=2):
            try:
                row = {key: (row.get(key) or "").strip() for key in _FIELDS | ({"threshold_approved", "mask_threshold"} & row.keys())}
                label = _LABELS.get(row["label"].lower())
                if label is None:
                    raise ValueError("label must be Au/authentic or Tp/tampered")
                if not row["image_path"] or not row["source_group"]:
                    raise ValueError("image_path and source_group are required")
                image_path = (root / row["image_path"]).resolve()
                if image_path in seen_images:
                    raise ValueError(f"Duplicate image path: {image_path}")
                seen_images.add(image_path)
                mask_path = None
                if label == "authentic":
                    if row["mask_path"]:
                        raise ValueError("Authentic rows must leave mask_path empty")
                else:
                    if not row["mask_path"]:
                        raise ValueError("Tampered sample requires mask_path")
                    if row["mask_verified"].lower() != "true" or not row["mask_source"]:
                        raise ValueError("Tampered mask requires mask_verified=true and mask_source")
                    mask_path = (root / row["mask_path"]).resolve()
                    if mask_path == image_path or mask_path in seen_masks:
                        raise ValueError("Mask path is reused or identical to image path")
                    seen_masks.add(mask_path)
                rgb = _read_rgb(image_path)
                if mask_path is not None:
                    _read_mask(mask_path, rgb.size, **threshold_options(row))
                # Decoded-pixel hashing also catches identical images in different
                # lossless encodings; near-duplicates still need a curator audit.
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                prior = content_groups.get(digest)
                if prior is not None and prior != (row["source_group"], label):
                    raise ValueError("Duplicate image content has conflicting source_group or label")
                content_groups[digest] = (row["source_group"], label)
                records.append({**row, "label": label, "image_path": image_path,
                                "mask_path": mask_path, "image_hash": digest})
            except (OSError, ValueError) as error:
                raise ValueError(f"Manifest row {line}: {error}") from error
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        rgb = _read_rgb(record["image_path"])
        if self.perturbation is not None:
            perturbed = self.perturbation(rgb, str(record["image_path"]))
            if perturbed.size != rgb.size:
                raise ValueError("A perturbation must keep the image size")
            rgb = perturbed
        height, width = self.image_size
        image = np.array(rgb.resize((width, height), Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0
        if record["mask_path"] is None:
            mask = np.zeros((height, width), dtype=np.float32)
        else:
            original = _read_mask(record["mask_path"], rgb.size, **threshold_options(record))
            mask = np.array(original.resize((width, height), Image.Resampling.NEAREST),
                            dtype=np.float32)
            if not mask.any():
                raise ValueError(f"Resizing removed all foreground: {record['mask_path']}")
        return (torch.from_numpy(image.transpose(2, 0, 1).copy()),
                torch.from_numpy(mask[None].copy()))


def split_dataset(dataset: CASIADataset, ratios=(0.7, 0.15, 0.15), seed: int = 42
                  ) -> tuple[Subset, Subset, Subset]:
    """Seeded, disjoint source-group split, returning train/validation/test.

    Ratios approximate GROUP counts, not image counts; no class stratification.
    Every split gets at least one group. Audit resulting class/sample counts.
    Curators must assign a shared group to ALL related originals/donors/edits.
    """
    if len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("Provide three positive finite split ratios")
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1")
    groups = sorted({row["source_group"] for row in dataset.records})
    if len(groups) < 3:
        raise ValueError("At least three source groups are required")
    random.Random(seed).shuffle(groups)
    counts = [1, 1, 1]
    # Allocate remaining groups to the split furthest below its target.
    for _ in range(len(groups) - 3):
        i = max(range(3), key=lambda i: ratios[i] * len(groups) - counts[i])
        counts[i] += 1
    results, start = [], 0
    for count in counts:
        selected = set(groups[start:start + count])
        results.append(Subset(dataset, [i for i, row in enumerate(dataset.records)
                                        if row["source_group"] in selected]))
        start += count
    return tuple(results)


def validate_split_isolation(*datasets: CASIADataset) -> None:
    """Reject group, path or decoded-pixel overlap between separate manifests."""
    for field in ("source_group", "image_path", "image_hash", "mask_path"):
        seen = set()
        for dataset in datasets:
            current = {row[field] for row in dataset.records if row[field] is not None}
            if seen & current:
                raise ValueError(f"Cross-split overlap detected in {field}")
            seen.update(current)
