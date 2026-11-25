"""Predict Real/Fake, confidence, mask, boxes and suspicious area for images.

    python scripts/predict.py --checkpoint RUN/best_dice.pt --image photo.jpg \
        --output-dir outputs/predictions

Uses ``thresholds.json`` next to the checkpoint when present (written by
``scripts/evaluate.py --calibrate`` on validation); otherwise defaults of 0.5
are used and a warning is printed. Per image it writes ``<name>_mask.png``,
``<name>_overlay.png``, ``<name>_heatmap.png`` and ``<name>_report.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from forgery_localization.inference import (
    Predictor, load_thresholds, mask_image, render_overlay, report_text,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", nargs="+", required=True, help="Image files or folders")
    parser.add_argument("--thresholds-file", default=None)
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    thresholds_path = Path(args.thresholds_file) if args.thresholds_file else checkpoint.parent / "thresholds.json"
    if not thresholds_path.is_file():
        print(f"WARNING: {thresholds_path} not found; using uncalibrated default thresholds.")
        thresholds_path = None
    predictor = Predictor(checkpoint, thresholds=load_thresholds(thresholds_path), device=args.device)

    paths = []
    for item in map(Path, args.image):
        paths += sorted(p for p in item.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if item.is_dir() else [item]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with Image.open(path) as opened:
            image = opened.copy()
        probabilities, mask, summary = predictor.predict(image)
        stem = path.stem
        mask_image(mask).save(output / f"{stem}_mask.png")
        render_overlay(image, mask, summary).save(output / f"{stem}_overlay.png")
        Image.fromarray((probabilities * 255).astype(np.uint8)).save(output / f"{stem}_heatmap.png")
        report = {"image": str(path), "model": predictor.name, **summary.to_dict(),
                  "thresholds": predictor.thresholds}
        (output / f"{stem}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n{path.name}\n{report_text(summary, predictor.name)}")


if __name__ == "__main__":
    main()
