"""NumPy-only tests for Real/Fake decision, boxes, area and image-level metrics."""

import math

import numpy as np
import pytest

from forgery_localization.evaluation.image_metrics import (
    choose_threshold, classification_metrics, pixel_summary, roc_auc,
)
from forgery_localization.evaluation.postprocess import (
    connected_regions, image_score, summarize_prediction,
)


def test_connected_regions_boxes_are_inclusive_and_sorted():
    mask = np.zeros((10, 12), bool)
    mask[1:3, 1:4] = True          # 6 px
    mask[5:9, 6:11] = True         # 20 px
    mask[0, 11] = True             # 1 px
    regions = connected_regions(mask)
    assert [r["area_px"] for r in regions] == [20, 6, 1]
    assert regions[0] == dict(x_min=6, y_min=5, x_max=10, y_max=8, area_px=20)
    assert len(connected_regions(mask, min_area=2)) == 2


def test_diagonal_pixels_are_one_region():
    mask = np.eye(5, dtype=bool)
    assert len(connected_regions(mask)) == 1


def test_image_score_methods():
    probabilities = np.zeros((10, 10), np.float32)
    probabilities[0, 0] = 0.9
    assert image_score(probabilities, "max") == pytest.approx(0.9)
    assert image_score(probabilities, "top1pct_mean") == pytest.approx(0.9)  # k = 1 pixel
    assert image_score(probabilities, "area_fraction") == pytest.approx(0.01)
    with pytest.raises(ValueError):
        image_score(probabilities, "unknown")
    with pytest.raises(ValueError):
        image_score(probabilities + 2, "max")


def test_summary_forged_and_authentic():
    probabilities = np.zeros((100, 100), np.float32)
    probabilities[10:30, 40:60] = 0.95
    mask, summary = summarize_prediction(probabilities, image_threshold=0.5)
    assert summary.label == "Forged"
    assert summary.num_regions == 1
    assert summary.regions[0]["x_min"] == 40 and summary.regions[0]["y_max"] == 29
    assert summary.suspicious_area_pct == pytest.approx(4.0)
    assert mask.sum() == 400

    _, clean = summarize_prediction(np.full((50, 50), 0.1, np.float32), image_threshold=0.5)
    assert clean.label == "Authentic"
    assert clean.confidence == pytest.approx(0.9)
    assert clean.regions == []


def test_summary_accepts_external_score():
    probabilities = np.zeros((20, 20), np.float32)
    _, summary = summarize_prediction(probabilities, image_threshold=0.3, score=0.4)
    assert summary.label == "Forged" and summary.image_score == pytest.approx(0.4)


def test_roc_auc_and_ties():
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert roc_auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert math.isnan(roc_auc([0.2, 0.3], [1, 1]))


def test_classification_metrics_confusion_matrix():
    metrics = classification_metrics([0.9, 0.8, 0.2, 0.7, 0.1], [1, 1, 1, 0, 0], threshold=0.5)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (2, 1, 1, 1)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == pytest.approx(0.6)


def test_choose_threshold_separates_perfectly():
    threshold = choose_threshold([0.1, 0.2, 0.7, 0.9], [0, 0, 1, 1])
    assert 0.2 < threshold <= 0.7
    assert classification_metrics([0.1, 0.2, 0.7, 0.9], [0, 0, 1, 1], threshold)["f1"] == 1.0


def test_pixel_summary_micro_and_per_image():
    rows = [
        dict(tp=8, fp=2, fn=0, n_pixels=100, forged=True),   # Dice 0.8889
        dict(tp=0, fp=0, fn=10, n_pixels=100, forged=True),  # Dice 0
        dict(tp=0, fp=5, fn=0, n_pixels=100, forged=False),
        dict(tp=0, fp=0, fn=0, n_pixels=100, forged=False),
    ]
    summary = pixel_summary(rows)
    assert summary["micro_dice"] == pytest.approx(16 / (16 + 7 + 10))
    assert summary["mean_tampered_dice"] == pytest.approx((16 / 18) / 2)
    assert summary["authentic_fp_pixel_pct"] == pytest.approx(2.5)
    assert summary["authentic_any_fp_pct"] == pytest.approx(50.0)
