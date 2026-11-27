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
    edge[:, :20] = (220, 10, 10)             # host image, camera 1 (red)
    edge[:, 20:] = (0, 200, 0)               # pasted object, camera 2 (green)
    Image.fromarray(edge).save(root / "4cam_splc" / "edgemask" / "canong3_canonxt_sub_01_edgemask.png")

    mask = builder.columbia_mask(root / "4cam_splc" / "edgemask" / "canong3_canonxt_sub_01_edgemask.png")
    assert mask[:, 20:].all() and not mask[:, :20].any()

    rows, problems = builder.columbia_rows(root, tmp_path / "masks", "green")
    assert builder.columbia_mask(root / "4cam_splc" / "edgemask" / "canong3_canonxt_sub_01_edgemask.png",
                                 "red")[:, :20].all()
    assert not problems
    assert [r["label"] for r in rows] == ["Au", "Tp"]
    saved = np.asarray(Image.open(rows[1]["mask_path"]))
    assert saved.shape == (30, 40) and saved[:, 20:].min() == 255 and saved[:, :20].max() == 0


def test_imd2020_rows(tmp_path):
    builder = load_script("build_external_manifest")
    folder = tmp_path / "imd2020" / "c8tf5mq"
    folder.mkdir(parents=True)
    rgb_image().save(folder / "c8tf5mq_orig.jpg")
    rgb_image().save(folder / "c8tf5mq_0.png")
    rgb_image().save(folder / "c8tf5mq_1.jpg")           # no mask -> skipped
    mask = np.zeros((30, 40), np.uint8)
    mask[5:15, 10:30] = 250                              # near-white, binarised at 127
    Image.fromarray(mask).save(folder / "c8tf5mq_0_mask.png")

    rows, problems = builder.imd2020_rows(tmp_path / "imd2020", tmp_path / "masks")
    assert problems == ["No mask for c8tf5mq_1.jpg"]
    assert sorted(r["label"] for r in rows) == ["Au", "Tp"]
    assert {r["source_group"] for r in rows} == {"imd2020_c8tf5mq"}
    tampered = next(r for r in rows if r["label"] == "Tp")
    saved = np.asarray(Image.open(tampered["mask_path"]))
    assert saved[5:15, 10:30].min() == 255 and saved.sum() == 255 * 10 * 20


def test_nist16_rows_invert_black_masks(tmp_path):
    builder = load_script("build_external_manifest")
    root = tmp_path / "NC2016_Test0613"
    (root / "probe").mkdir(parents=True)
    (root / "mask").mkdir()
    (root / "reference" / "manipulation").mkdir(parents=True)
    rgb_image().save(root / "probe" / "NC2016_0001.jpg")
    rgb_image().save(root / "probe" / "NC2016_0002.jpg")
    nist_mask = np.full((30, 40), 255, np.uint8)
    nist_mask[:, :10] = 0                                # black = manipulated in NIST
    Image.fromarray(nist_mask).save(root / "mask" / "NC2016_0001.png")
    reference = root / "reference" / "manipulation" / "NC2016-manipulation-ref.csv"
    reference.write_text(
        "TaskID|ProbeFileID|ProbeFileName|ProbeMaskFileName|BaseFileName|IsTarget\n"
        "manipulation|NC2016_0001|probe/NC2016_0001.jpg|mask/NC2016_0001.png|world/base_7.jpg|Y\n"
        "manipulation|NC2016_0002|probe/NC2016_0002.jpg||base_7.jpg|N\n", encoding="utf-8")

    rows, problems = builder.nist16_rows(root, reference, tmp_path / "masks")
    assert not problems
    assert [r["label"] for r in rows] == ["Tp", "Au"]
    assert {r["source_group"] for r in rows} == {"nist16_base_7"}
    saved = np.asarray(Image.open(rows[0]["mask_path"]))
    assert saved[:, :10].min() == 255 and saved[:, 10:].max() == 0


def test_build_manifest_cli_nist16(tmp_path):
    builder = load_script("build_external_manifest")
    root = tmp_path / "nist"
    (root / "probe").mkdir(parents=True)
    rgb_image().save(root / "probe" / "a.jpg")
    ref = tmp_path / "ref.csv"
    ref.write_text("ProbeFileName|ProbeMaskFileName|IsTarget\nprobe/a.jpg||N\n", encoding="utf-8")
    output = tmp_path / "out" / "nist16_test.csv"
    builder.main(["nist16", "--dataset-root", str(root), "--nist-ref", str(ref), "--output", str(output)])
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["label"] == "Au" and rows[0]["mask_path"] == ""
    with pytest.raises(SystemExit):
        builder.main(["imd2020", "--output", str(output)])
