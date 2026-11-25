"""Build a CASIADataset-compatible manifest for an external (cross-dataset) test set.

Presets
-------
coverage  COVERAGE copy-move dataset (Wen et al., 2016) laid out as
          ``<root>/image/{n}.tif`` (authentic), ``<root>/image/{n}t.tif``
          (tampered) and ``<root>/mask/{n}forged.tif`` (tampered region).
generic   Any dataset with an authentic folder, a tampered folder and a mask
          folder; masks are found with ``--mask-pattern`` (``{stem}`` = image
          file name without extension), e.g. ``{stem}_gt.png``.

Masks are only marked ``mask_verified=true`` when ``--masks-checked`` is given.
Run once without it, open the preview image, confirm that WHITE = manipulated
region and that masks align with the images, then rerun with --masks-checked.
External data is used for testing only: never train or calibrate on it.

    python scripts/build_external_manifest.py coverage --dataset-root /content/COVERAGE \
        --output data/splits/coverage_test.csv --preview reports/coverage_preview.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
FIELDS = ["image_path", "mask_path", "label", "source_group", "mask_verified", "mask_source",
          "threshold_approved", "mask_threshold"]


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def coverage_rows(root: Path) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for path in _images(root / "image"):
        match = re.fullmatch(r"(\d+)(t?)", path.stem)
        if not match:
            problems.append(f"Unrecognised file name: {path.name}")
            continue
        number, tampered = match.groups()
        group = f"coverage_{number}"
        if not tampered:
            rows.append({"image_path": path, "mask_path": "", "label": "Au", "source_group": group})
            continue
        candidates = list((root / "mask").glob(f"{number}forged.*"))
        if len(candidates) != 1:
            problems.append(f"No unique mask for {path.name}")
            continue
        rows.append({"image_path": path, "mask_path": candidates[0], "label": "Tp", "source_group": group,
                     "mask_source": "COVERAGE mask/{n}forged"})
    return rows, problems


def generic_rows(authentic_dir: Path | None, tampered_dir: Path, mask_dir: Path, pattern: str,
                 group_regex: str | None, name: str) -> tuple[list[dict], list[str]]:
    rows, problems = [], []

    def group(stem):
        if group_regex:
            match = re.search(group_regex, stem)
            if match:
                return f"{name}_{match.group(1)}"
        return f"{name}_{stem}"

    if authentic_dir is not None:
        rows += [{"image_path": p, "mask_path": "", "label": "Au", "source_group": group(p.stem)}
                 for p in _images(authentic_dir)]
    for path in _images(tampered_dir):
        mask = mask_dir / pattern.format(stem=path.stem)
        if not mask.is_file():
            problems.append(f"Missing mask {mask.name} for {path.name}")
            continue
        rows.append({"image_path": path, "mask_path": mask, "label": "Tp", "source_group": group(path.stem),
                     "mask_source": f"{name} {mask_dir.name}/{pattern}"})
    return rows, problems


def check_pair(row: dict) -> str | None:
    if not row["mask_path"]:
        return None
    with Image.open(row["image_path"]) as image, Image.open(row["mask_path"]) as mask:
        if image.size != mask.size:
            return f"size mismatch {image.size} vs {mask.size}: {Path(row['image_path']).name}"
    return None


def write_preview(rows: list[dict], path: Path, count: int, threshold: int | None) -> None:
    tampered = [r for r in rows if r["mask_path"]][:count]
    if not tampered:
        return
    tiles = []
    for row in tampered:
        with Image.open(row["image_path"]) as image, Image.open(row["mask_path"]) as mask:
            rgb = image.convert("RGB").resize((192, 192))
            values = np.asarray(mask.convert("L").resize((192, 192), Image.Resampling.NEAREST))
        binary = values > (threshold if threshold is not None else 127)
        overlay = np.asarray(rgb, dtype=np.float32)
        overlay[binary] = 0.5 * overlay[binary] + 0.5 * np.array([255, 0, 0])
        tile = Image.new("RGB", (192 * 3 + 8, 192), "white")
        tile.paste(rgb, (0, 0))
        tile.paste(Image.fromarray(values).convert("RGB"), (196, 0))
        tile.paste(Image.fromarray(overlay.astype(np.uint8)), (392, 0))
        tiles.append(tile)
    canvas = Image.new("RGB", (tiles[0].width, len(tiles) * 196), "white")
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (0, i * 196))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    print("Preview (image | mask | red = mask > threshold):", path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preset", choices=("coverage", "generic"))
    parser.add_argument("--dataset-root", help="coverage: folder containing image/ and mask/")
    parser.add_argument("--authentic-dir")
    parser.add_argument("--tampered-dir")
    parser.add_argument("--mask-dir")
    parser.add_argument("--mask-pattern", default="{stem}_gt.png")
    parser.add_argument("--group-regex", default=None, help="Regex whose group(1) defines source_group")
    parser.add_argument("--name", default="external")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-threshold", type=int, default=None,
                        help="Approve thresholding of non-binary masks at this value (0-254)")
    parser.add_argument("--masks-checked", action="store_true",
                        help="I inspected the preview: masks align and white = manipulated")
    parser.add_argument("--preview", default=None, help="PNG path for a visual mask check")
    parser.add_argument("--preview-count", type=int, default=12)
    args = parser.parse_args(argv)

    if args.preset == "coverage":
        if not args.dataset_root:
            parser.error("coverage requires --dataset-root")
        rows, problems = coverage_rows(Path(args.dataset_root))
    else:
        if not (args.tampered_dir and args.mask_dir):
            parser.error("generic requires --tampered-dir and --mask-dir")
        rows, problems = generic_rows(Path(args.authentic_dir) if args.authentic_dir else None,
                                      Path(args.tampered_dir), Path(args.mask_dir), args.mask_pattern,
                                      args.group_regex, args.name)
    problems += [p for p in map(check_pair, rows) if p]
    bad = {Path(p.split(": ")[-1]).name for p in problems}
    rows = [r for r in rows if Path(r["image_path"]).name not in bad]
    for row in rows:
        row["image_path"] = str(Path(row["image_path"]).resolve())
        row["mask_path"] = str(Path(row["mask_path"]).resolve()) if row["mask_path"] else ""
        tampered = row["label"] == "Tp"
        row["mask_verified"] = "true" if tampered and args.masks_checked else ""
        row.setdefault("mask_source", "")
        row["threshold_approved"] = "true" if tampered and args.mask_threshold is not None else ""
        row["mask_threshold"] = str(args.mask_threshold) if tampered and args.mask_threshold is not None else ""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n_tp = sum(r["label"] == "Tp" for r in rows)
    print(f"Wrote {output}: {len(rows)} rows ({len(rows) - n_tp} authentic, {n_tp} tampered)")
    for problem in problems:
        print("SKIPPED:", problem)
    if args.preview:
        write_preview(rows, Path(args.preview), args.preview_count, args.mask_threshold)
    if not args.masks_checked:
        print("Masks NOT marked verified. Inspect the preview, then rerun with --masks-checked "
              "(the loader rejects unverified tampered masks).")
    print("Use relative-free paths: pass --root / to evaluate.py (paths are absolute).")


if __name__ == "__main__":
    main()
