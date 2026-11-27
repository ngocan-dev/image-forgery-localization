"""Evaluate trained checkpoints on a manifest (CASIA val/test or an external dataset).

Two-step protocol (no test-set tuning):

1. Calibrate on VALIDATION -- choose the image-level Real/Fake threshold and
   save ``thresholds.json`` next to each checkpoint::

       python scripts/evaluate.py --checkpoint RUN/best_dice.pt --manifest val.csv \
           --dataset-name casia_val --calibrate --output-dir reports/eval

2. Evaluate TEST / external datasets with the frozen thresholds::

       python scripts/evaluate.py --checkpoint RUN/best_dice.pt --manifest test.csv \
           --dataset-name casia_test --output-dir reports/eval --save-visuals 6

Pixel metrics are computed at model resolution (256 x 256) against the
nearest-neighbour resized ground truth, exactly as during training.
Outputs per run: ``per_image.csv``, ``summary.json`` and optional figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from forgery_localization.datasets.casia_dataset import CASIADataset, validate_split_isolation
from forgery_localization.evaluation.image_metrics import (
    choose_threshold, classification_metrics, pixel_summary,
)
from forgery_localization.evaluation.postprocess import SCORE_METHODS, image_score
from forgery_localization.inference import CLS_SCORE, Predictor, render_overlay
from forgery_localization.models.factory import model_label
from forgery_localization.perturbations import parse_perturbation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", nargs="+", required=True, help="One or more best_dice.pt files")
    parser.add_argument("--manifest", required=True, help="CSV manifest (same format as training)")
    parser.add_argument("--root", default=".", help="Directory that relative manifest paths resolve against")
    parser.add_argument("--dataset-name", required=True, help="Label used in reports, e.g. casia_test")
    parser.add_argument("--output-dir", default="reports/eval")
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pixel-threshold", type=float, default=0.5)
    parser.add_argument("--score-method", choices=("auto", CLS_SCORE, *SCORE_METHODS), default="auto",
                        help="Image score for Real/Fake; auto = 'cls' (classification head) when the "
                             "checkpoint has one, otherwise top1pct_mean")
    parser.add_argument("--calibrate", action="store_true",
                        help="Choose the image threshold on THIS manifest (use only for validation)")
    parser.add_argument("--criterion", choices=("f1", "balanced_accuracy"), default="f1")
    parser.add_argument("--thresholds-file", default=None,
                        help="Override; default is thresholds.json next to each checkpoint")
    parser.add_argument("--isolation-manifests", nargs="*", default=[],
                        help="Train/val manifests that must not overlap with --manifest")
    parser.add_argument("--perturb", default="none",
                        help="Robustness test: jpeg:Q, resize:F, noise:S or blur:R (applied before resizing)")
    parser.add_argument("--save-visuals", type=int, default=0,
                        help="Save N best, N worst tampered and N worst authentic examples")
    return parser.parse_args(argv)


def run_name(config: dict) -> str:
    return f"{model_label(config)}_{config['mode']}_seed{config.get('seed', 'na')}"


def resolve_score_method(requested: str, has_cls_head: bool) -> str:
    if requested == "auto":
        return CLS_SCORE if has_cls_head else "top1pct_mean"
    if requested == CLS_SCORE and not has_cls_head:
        raise SystemExit("--score-method cls needs a checkpoint trained with --cls-head")
    return requested


def collect_predictions(predictor: Predictor, dataset: CASIADataset, batch_size: int,
                        pixel_threshold: float) -> tuple[list[dict], dict[int, np.ndarray]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    rows, probability_maps, index = [], {}, 0
    for images, masks in loader:
        probabilities, cls = predictor.forward(images)
        probabilities = probabilities[:, 0].numpy()
        cls = [None] * len(probabilities) if cls is None else cls.tolist()
        truth = masks[:, 0].numpy() >= 0.5
        for probability, target, cls_score in zip(probabilities, truth, cls):
            record = dataset.records[index]
            predicted = probability >= pixel_threshold
            tp = int((predicted & target).sum())
            fp = int((predicted & ~target).sum())
            fn = int((~predicted & target).sum())
            forged = record["label"] == "tampered"
            if forged != bool(target.any()):
                raise ValueError(f"Label and mask disagree for {record['image_path']}")
            rows.append({
                "index": index,
                "image_path": str(record["image_path"]),
                "label": record["label"],
                "forged": forged,
                "tp": tp, "fp": fp, "fn": fn,
                "n_pixels": int(target.size),
                "dice": 2 * tp / max(2 * tp + fp + fn, 1),
                "iou": tp / max(tp + fp + fn, 1),
                "gt_area_pct": 100 * float(target.mean()),
                "pred_area_pct": 100 * float(predicted.mean()),
                **{f"score_{m}": image_score(probability, m, pixel_threshold) for m in SCORE_METHODS},
                **({} if cls_score is None else {f"score_{CLS_SCORE}": cls_score}),
            })
            probability_maps[index] = probability
            index += 1
    return rows, probability_maps


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    return value


def save_visuals(rows, probability_maps, dataset, pixel_threshold, count, directory: Path) -> None:
    tampered = sorted((r for r in rows if r["forged"]), key=lambda r: r["dice"])
    authentic = sorted((r for r in rows if not r["forged"]), key=lambda r: -r["fp"])
    groups = {"best_tampered": tampered[::-1][:count], "worst_tampered": tampered[:count],
              "worst_authentic_fp": [r for r in authentic[:count] if r["fp"] > 0]}
    for group, selected in groups.items():
        folder = directory / group
        folder.mkdir(parents=True, exist_ok=True)
        for row in selected:
            image, mask = dataset[row["index"]]
            rgb = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
            gt = mask[0].numpy() >= 0.5
            predicted = probability_maps[row["index"]] >= pixel_threshold
            heat = Image.fromarray((probability_maps[row["index"]] * 255).astype(np.uint8)).convert("RGB")
            panels = [rgb, render_overlay(rgb, gt), heat, render_overlay(rgb, predicted)]
            canvas = Image.new("RGB", (rgb.width * 4 + 30, rgb.height), "white")
            for i, panel in enumerate(panels):
                canvas.paste(panel, (i * (rgb.width + 10), 0))
            name = Path(row["image_path"]).stem
            canvas.save(folder / f"{name}_dice{row['dice']:.3f}_fp{row['fp']}.png")
    (directory / "README.txt").write_text(
        "Panels: image | ground truth | probability map | prediction (threshold "
        f"{pixel_threshold}). Selection: Dice ranking for tampered images, FP pixel count for authentic.\n",
        encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.calibrate and "test" in args.dataset_name.lower():
        raise SystemExit("Refusing to calibrate thresholds on a test set; calibrate on validation.")
    perturbation = parse_perturbation(args.perturb)
    if args.calibrate and perturbation is not None:
        raise SystemExit("Calibrate on clean validation images only (no --perturb).")
    dataset = CASIADataset(args.manifest, root=args.root, image_size=tuple(args.image_size),
                           perturbation=perturbation)
    if args.isolation_manifests:
        others = [CASIADataset(m, root=args.root, image_size=tuple(args.image_size))
                  for m in args.isolation_manifests]
        validate_split_isolation(*others, dataset)
        print("Isolation check passed against", len(others), "manifest(s)")
    output_root = Path(args.output_dir) / args.dataset_name
    combined = []
    for checkpoint in args.checkpoint:
        checkpoint = Path(checkpoint)
        thresholds_path = Path(args.thresholds_file) if args.thresholds_file else checkpoint.parent / "thresholds.json"
        predictor = Predictor(checkpoint, device=args.device)
        name = run_name(predictor.config)
        score_method = resolve_score_method(args.score_method, predictor.has_cls_head)
        print(f"\n== {name} on {args.dataset_name} ({len(dataset)} images, image score: {score_method})")
        rows, probability_maps = collect_predictions(predictor, dataset, args.batch_size, args.pixel_threshold)
        labels = [row["forged"] for row in rows]
        scores = [row[f"score_{score_method}"] for row in rows]

        if args.calibrate:
            threshold = choose_threshold(scores, labels, args.criterion)
            thresholds = {"pixel_threshold": args.pixel_threshold, "image_threshold": threshold,
                          "score_method": score_method, "criterion": args.criterion,
                          "calibrated_on": args.dataset_name}
            thresholds_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
            print("Saved thresholds:", thresholds_path)
        else:
            if not thresholds_path.is_file():
                raise SystemExit(f"Missing {thresholds_path}; run --calibrate on validation first.")
            thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
            if thresholds["score_method"] != score_method or thresholds["pixel_threshold"] != args.pixel_threshold:
                raise SystemExit("Thresholds file was calibrated with a different score method/pixel threshold")

        image_level = classification_metrics(scores, labels, thresholds["image_threshold"])
        for row, score in zip(rows, scores):
            row["predicted_label"] = "tampered" if score >= thresholds["image_threshold"] else "authentic"
        auc_by_method = {}
        for method in SCORE_METHODS + ((CLS_SCORE,) if predictor.has_cls_head else ()):
            method_scores = [row[f"score_{method}"] for row in rows]
            auc_by_method[method] = classification_metrics(method_scores, labels, 0.5)["auc"]
        summary = {
            "run": name,
            "dataset": args.dataset_name,
            "architecture": model_label(predictor.config),
            "mode": predictor.config["mode"],
            "seed": predictor.config.get("seed"),
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": predictor.config.get("checkpoint_epoch"),
            "perturbation": args.perturb,
            "thresholds": thresholds,
            "pixel": pixel_summary(rows),
            "image": image_level,
            "image_auc_by_score_method": auc_by_method,
        }
        directory = output_root / name
        directory.mkdir(parents=True, exist_ok=True)
        write_csv(rows, directory / "per_image.csv")
        (directory / "summary.json").write_text(json.dumps(_clean(summary), indent=2), encoding="utf-8")
        if args.save_visuals:
            save_visuals(rows, probability_maps, dataset, args.pixel_threshold, args.save_visuals,
                         directory / "visuals")
        pixel, image = summary["pixel"], summary["image"]
        print(f"Pixel  micro Dice {pixel['micro_dice']:.4f} | micro IoU {pixel['micro_iou']:.4f} | "
              f"mean tampered Dice {pixel['mean_tampered_dice']:.4f} | "
              f"authentic FP pixels {pixel['authentic_fp_pixel_pct']:.3f}%")
        print(f"Image  Acc {image['accuracy']:.4f} | P {image['precision']:.4f} | R {image['recall']:.4f} | "
              f"F1 {image['f1']:.4f} | AUC {image['auc']:.4f} | "
              f"CM [[TN {image['tn']}, FP {image['fp']}], [FN {image['fn']}, TP {image['tp']}]]")
        combined.append({"run": name, "architecture": summary["architecture"], "mode": summary["mode"],
                         "seed": summary["seed"],
                         **{f"pixel_{k}": v for k, v in pixel.items()},
                         **{f"image_{k}": v for k, v in image.items()}})
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if combined:
        write_csv(combined, output_root / "runs_this_call.csv")


if __name__ == "__main__":
    main()
