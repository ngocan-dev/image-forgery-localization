"""Tests for the no-training analyses: score methods, perturbations, score study, Columbia masks."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forgery_localization.evaluation.postprocess import SCORE_METHODS, image_score
from forgery_localization.perturbations import parse_perturbation, perturbation_tag

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_score_methods():
    probabilities = np.zeros((100, 100), np.float32)
    probabilities[:10, :10] = 1.0            # 1% of the pixels
    assert image_score(probabilities, "top01pct_mean") == pytest.approx(1.0)
    assert image_score(probabilities, "top1pct_mean") == pytest.approx(1.0)
    assert image_score(probabilities, "top5pct_mean") == pytest.approx(0.2)
    assert image_score(probabilities, "mean") == pytest.approx(0.01)
    assert {"max", "top1pct_mean", "area_fraction"} <= set(SCORE_METHODS)


def rgb_image(width=40, height=30):
    pixels = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


@pytest.mark.parametrize("spec", ["jpeg:70", "resize:0.5", "noise:5", "blur:1"])
def test_perturbations_keep_size_and_change_pixels(spec):
    image = rgb_image()
    result = parse_perturbation(spec)(image, "key")
    assert result.size == image.size and result.mode == "RGB"
    assert not np.array_equal(np.asarray(result), np.asarray(image))


def test_noise_is_deterministic_per_key():
    image = rgb_image()
    noise = parse_perturbation("noise:8")
    assert np.array_equal(np.asarray(noise(image, "a")), np.asarray(noise(image, "a")))
    assert not np.array_equal(np.asarray(noise(image, "a")), np.asarray(noise(image, "b")))


def test_perturbation_parsing():
    assert parse_perturbation("none") is None
    assert perturbation_tag("resize:0.5") == "resize0p5" and perturbation_tag("jpeg:70") == "jpeg70"
    for bad in ("jpeg:0", "resize:1.5", "sharpen:2", "jpeg"):
        with pytest.raises(ValueError):
            parse_perturbation(bad)


def write_per_image(path, scores_by_method, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "forged", *[f"score_{m}" for m in scores_by_method]])
        for i, label in enumerate(labels):
            writer.writerow([i, label, *[scores[i] for scores in scores_by_method.values()]])


def test_score_study_selects_on_validation(tmp_path):
    study = load_script("score_study")
    labels = [False, False, True, True]
    for seed in (7, 21):
        run = f"unet_resnet34_rgb_seed{seed}"
        # "good" separates the classes on validation, "bad" does not
        write_per_image(tmp_path / "casia_val" / run / "per_image.csv",
                        {"good": [0.1, 0.2, 0.8, 0.9], "bad": [0.9, 0.1, 0.2, 0.8]}, labels)
        write_per_image(tmp_path / "casia_test" / run / "per_image.csv",
                        {"good": [0.1, 0.3, 0.85, 0.9], "bad": [0.5, 0.5, 0.5, 0.5]}, labels)
    rows = study.study(tmp_path, "casia_val", ["casia_test"], "f1")
    table = study.aggregate(rows)
    assert study.select_methods(table, "casia_val") == {("unet_resnet34", "rgb"): "good"}
    good_test = next(e for e in table if e["dataset"] == "casia_test" and e["method"] == "good")
    assert good_test["f1_mean"] == pytest.approx(1.0) and good_test["n"] == 2
    assert study.parse_run("segformer_mit_b2_rgb_seed42") == ("segformer_mit_b2", "rgb", "42")
    with pytest.raises(SystemExit):
        study.main(["--eval-dir", str(tmp_path), "--calibration", "casia_test", "--output-dir", str(tmp_path)])


def test_columbia_mask_and_rows(tmp_path):
    builder = load_script("build_external_manifest")
    root = tmp_path / "columbia"
    (root / "4cam_auth").mkdir(parents=True)
    (root / "4cam_splc" / "edgemask").mkdir(parents=True)
    rgb_image().save(root / "4cam_auth" / "canong3_02_sub_01.tif")
    rgb_image().save(root / "4cam_splc" / "canong3_canonxt_sub_01.tif")
    edge = np.zeros((30, 40, 3), np.uint8)
    edge[:, :20] = (0, 200, 0)               # authentic region (green)
    edge[:, 20:] = (220, 10, 10)             # spliced region (red)
    Image.fromarray(edge).save(root / "4cam_splc" / "edgemask" / "canong3_canonxt_sub_01_edgemask.png")

    mask = builder.columbia_mask(root / "4cam_splc" / "edgemask" / "canong3_canonxt_sub_01_edgemask.png")
    assert mask[:, 20:].all() and not mask[:, :20].any()

    rows, problems = builder.columbia_rows(root, tmp_path / "masks", "red")
    assert not problems
    assert [r["label"] for r in rows] == ["Au", "Tp"]
    saved = np.asarray(Image.open(rows[1]["mask_path"]))
    assert saved.shape == (30, 40) and saved[:, 20:].min() == 255 and saved[:, :20].max() == 0
