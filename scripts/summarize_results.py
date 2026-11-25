"""Aggregate evaluate.py outputs into thesis tables (mean +/- sample SD over seeds).

    python scripts/summarize_results.py --eval-dir reports/eval --output-dir reports/tables

Writes:
  all_runs.csv          one row per (dataset, run)
  mean_sd.csv / .md     mean +/- SD per (dataset, architecture, mode)
  paired_vs_rgb.csv     same-seed difference to the RGB run of the same architecture
  generalization.md     Dataset x configuration matrix of micro Dice and image F1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = [
    ("pixel", "micro_dice"), ("pixel", "micro_iou"), ("pixel", "mean_tampered_dice"),
    ("pixel", "authentic_fp_pixel_pct"), ("image", "accuracy"), ("image", "precision"),
    ("image", "recall"), ("image", "f1"), ("image", "auc"),
]


def _value(summary, section, key):
    value = summary[section].get(key)
    return float("nan") if value is None else float(value)


def _mean_sd(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else float("nan"))


def _fmt(mean, sd):
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.4f}" if math.isnan(sd) else f"{mean:.4f} ± {sd:.4f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="reports/eval")
    parser.add_argument("--output-dir", default="reports/tables")
    args = parser.parse_args(argv)
    summaries = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(Path(args.eval_dir).glob("*/*/summary.json"))]
    if not summaries:
        raise SystemExit(f"No summary.json files under {args.eval_dir}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    runs = [{"dataset": s["dataset"], "architecture": s["architecture"], "mode": s["mode"],
             "seed": s["seed"], **{f"{sec}_{key}": _value(s, sec, key) for sec, key in METRICS}}
            for s in summaries]
    with (output / "all_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)

    groups = defaultdict(list)
    for run in runs:
        groups[(run["dataset"], run["architecture"], run["mode"])].append(run)
    table = []
    for (dataset, architecture, mode), members in sorted(groups.items()):
        row = {"dataset": dataset, "architecture": architecture, "mode": mode,
               "n_seeds": len(members), "seeds": " ".join(str(m["seed"]) for m in members)}
        for sec, key in METRICS:
            mean, sd = _mean_sd([m[f"{sec}_{key}"] for m in members])
            row[f"{sec}_{key}_mean"], row[f"{sec}_{key}_sd"] = mean, sd
        table.append(row)
    with (output / "mean_sd.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    headers = ["Dataset", "Model", "Input", "n", "micro Dice", "micro IoU", "mean tampered Dice",
               "Auth. FP px %", "Img Acc", "Img F1", "Img AUC"]
    keys = ["pixel_micro_dice", "pixel_micro_iou", "pixel_mean_tampered_dice",
            "pixel_authentic_fp_pixel_pct", "image_accuracy", "image_f1", "image_auc"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in table:
        cells = [row["dataset"], row["architecture"], row["mode"], str(row["n_seeds"])]
        cells += [_fmt(row[f"{k}_mean"], row[f"{k}_sd"]) for k in keys]
        lines.append("| " + " | ".join(cells) + " |")
    (output / "mean_sd.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_key = {(r["dataset"], r["architecture"], r["mode"], r["seed"]): r for r in runs}
    paired = []
    for (dataset, architecture, mode, seed), run in sorted(by_key.items(), key=str):
        base = by_key.get((dataset, architecture, "rgb", seed))
        if mode == "rgb" or base is None:
            continue
        paired.append({"dataset": dataset, "architecture": architecture, "mode": mode, "seed": seed,
                       **{f"delta_{sec}_{key}": run[f"{sec}_{key}"] - base[f"{sec}_{key}"]
                          for sec, key in METRICS}})
    if paired:
        with (output / "paired_vs_rgb.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)

    datasets = sorted({r["dataset"] for r in table})
    configs = sorted({(r["architecture"], r["mode"]) for r in table})
    lookup = {(r["dataset"], r["architecture"], r["mode"]): r for r in table}
    matrix = ["Cells: micro Dice / image F1 (mean over seeds). Train = CASIA v2 for every row.", "",
              "| Model | Input | " + " | ".join(datasets) + " |", "|---|---|" + "---|" * len(datasets)]
    for architecture, mode in configs:
        cells = []
        for dataset in datasets:
            row = lookup.get((dataset, architecture, mode))
            cells.append("—" if row is None else
                         f"{row['pixel_micro_dice_mean']:.3f} / {row['image_f1_mean']:.3f}")
        matrix.append(f"| {architecture} | {mode} | " + " | ".join(cells) + " |")
    (output / "generalization.md").write_text("\n".join(matrix) + "\n", encoding="utf-8")
    print((output / "mean_sd.md").read_text(encoding="utf-8"))
    print((output / "generalization.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
