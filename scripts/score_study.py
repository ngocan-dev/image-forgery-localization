"""Compare image-level (Real/Fake) score methods without retraining.

Reads the ``per_image.csv`` files written by ``evaluate.py``. For every run
and every ``score_<method>`` column it chooses the threshold on the
calibration set (validation, max F1) and applies it unchanged to each target
dataset. The method recommended for each configuration is chosen on
validation only (highest mean validation F1 over seeds, ties -> higher worst
seed), so the test sets are never used for selection.

    python scripts/score_study.py --eval-dir reports/eval --calibration casia_val \
        --datasets casia_test coverage --output-dir reports/score_study

Writes ``per_run.csv`` (run x method x dataset), ``mean_sd.csv`` and
``score_study.md`` (mean +/- SD over seeds and the validation-selected method).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgery_localization.evaluation.image_metrics import choose_threshold, classification_metrics  # noqa: E402

METRICS = ("f1", "balanced_accuracy", "accuracy", "specificity", "recall", "auc")


def read_scores(path: Path) -> tuple[list[bool], dict[str, list[float]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    labels = [row["forged"].strip().lower() == "true" for row in rows]
    methods = {key[len("score_"):]: [float(row[key]) for row in rows]
               for key in rows[0] if key.startswith("score_")}
    return labels, methods


def parse_run(name: str) -> tuple[str, str, str]:
    stem, _, seed = name.rpartition("_seed")
    architecture, _, mode = stem.rpartition("_")
    return architecture, mode, seed


def mean_sd(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else math.nan


def fmt(mean: float, sd: float) -> str:
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.3f}" if math.isnan(sd) else f"{mean:.3f} ± {sd:.3f}"


def study(eval_dir: Path, calibration: str, datasets: list[str], criterion: str) -> list[dict]:
    rows = []
    for calibration_csv in sorted((eval_dir / calibration).glob("*/per_image.csv")):
        run = calibration_csv.parent.name
        architecture, mode, seed = parse_run(run)
        val_labels, val_scores = read_scores(calibration_csv)
        targets = {}
        for dataset in datasets:
            target_csv = eval_dir / dataset / run / "per_image.csv"
            if target_csv.is_file():
                targets[dataset] = read_scores(target_csv)
        for method, scores in val_scores.items():
            threshold = choose_threshold(scores, val_labels, criterion)
            val_metrics = classification_metrics(scores, val_labels, threshold)
            base = dict(run=run, architecture=architecture, mode=mode, seed=seed, method=method,
                        threshold=threshold, val_f1=val_metrics["f1"], val_auc=val_metrics["auc"])
            rows.append({**base, "dataset": calibration, **{m: val_metrics[m] for m in METRICS}})
            for dataset, (labels, methods) in targets.items():
                if method not in methods:
                    continue          # older per_image.csv without this score column
                metrics = classification_metrics(methods[method], labels, threshold)
                rows.append({**base, "dataset": dataset, **{m: metrics[m] for m in METRICS}})
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["architecture"], row["mode"], row["method"])].append(row)
    table = []
    for (dataset, architecture, mode, method), members in sorted(groups.items()):
        entry = dict(dataset=dataset, architecture=architecture, mode=mode, method=method, n=len(members))
        for metric in ("threshold", *METRICS):
            entry[f"{metric}_mean"], entry[f"{metric}_sd"] = mean_sd([float(r[metric]) for r in members])
        entry["val_f1_mean"], _ = mean_sd([float(r["val_f1"]) for r in members])
        entry["val_f1_min"] = min(float(r["val_f1"]) for r in members)
        table.append(entry)
    return table


def select_methods(table: list[dict], calibration: str) -> dict[tuple[str, str], str]:
    best = {}
    for entry in table:
        if entry["dataset"] != calibration:
            continue
        key = (entry["architecture"], entry["mode"])
        rank = (entry["val_f1_mean"], entry["val_f1_min"])
        if key not in best or rank > best[key][0]:
            best[key] = (rank, entry["method"])
    return {key: method for key, (_, method) in best.items()}


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(table: list[dict], selected: dict, calibration: str, datasets: list[str], default: str) -> str:
    lines = [f"Threshold chosen on `{calibration}` for every run and score method, then frozen. "
             "Mean ± SD over seeds.", ""]
    lines += ["### Validation-selected score method per configuration", "",
              "| Model | Input | Selected method | Dataset | F1 | Balanced acc. | Specificity | AUC | "
              f"F1 with `{default}` |", "|---|---|---|---|---|---|---|---|---|"]
    index = {(e["dataset"], e["architecture"], e["mode"], e["method"]): e for e in table}
    for (architecture, mode), method in sorted(selected.items()):
        for dataset in (calibration, *datasets):
            entry = index.get((dataset, architecture, mode, method))
            if entry is None:
                continue
            reference = index.get((dataset, architecture, mode, default))
            lines.append(
                f"| {architecture} | {mode} | {method} | {dataset} | "
                f"{fmt(entry['f1_mean'], entry['f1_sd'])} | "
                f"{fmt(entry['balanced_accuracy_mean'], entry['balanced_accuracy_sd'])} | "
                f"{fmt(entry['specificity_mean'], entry['specificity_sd'])} | "
                f"{fmt(entry['auc_mean'], entry['auc_sd'])} | "
                f"{fmt(reference['f1_mean'], reference['f1_sd']) if reference else 'n/a'} |")
    for dataset in (calibration, *datasets):
        members = [e for e in table if e["dataset"] == dataset]
        if not members:
            continue
        lines += ["", f"### All score methods on `{dataset}`", "",
                  "| Model | Input | Method | Threshold | F1 | Balanced acc. | AUC |", "|---|---|---|---|---|---|---|"]
        for e in members:
            lines.append(f"| {e['architecture']} | {e['mode']} | {e['method']} | "
                         f"{fmt(e['threshold_mean'], e['threshold_sd'])} | {fmt(e['f1_mean'], e['f1_sd'])} | "
                         f"{fmt(e['balanced_accuracy_mean'], e['balanced_accuracy_sd'])} | "
                         f"{fmt(e['auc_mean'], e['auc_sd'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--calibration", default="casia_val")
    parser.add_argument("--datasets", nargs="*", default=["casia_test"])
    parser.add_argument("--criterion", choices=("f1", "balanced_accuracy"), default="f1")
    parser.add_argument("--default-method", default="top1pct_mean")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if "test" in args.calibration.lower():
        raise SystemExit("The calibration set must not be a test set.")

    rows = study(Path(args.eval_dir), args.calibration, args.datasets, args.criterion)
    if not rows:
        raise SystemExit(f"No per_image.csv found under {Path(args.eval_dir) / args.calibration}")
    table = aggregate(rows)
    selected = select_methods(table, args.calibration)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "per_run.csv")
    write_csv(table, output / "mean_sd.csv")
    text = markdown(table, selected, args.calibration, args.datasets, args.default_method)
    (output / "score_study.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
