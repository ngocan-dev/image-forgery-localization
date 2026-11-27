"""Feature extraction, model factory and checkpoint round-trip (needs torch + smp)."""

import json
import math

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from forgery_localization.features import FeatureBuilder, input_channels  # noqa: E402
from forgery_localization.inference import Predictor, render_overlay  # noqa: E402
from forgery_localization.models.factory import build_model, load_checkpoint  # noqa: E402


def _image(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, 64, 64, generator=generator)


def test_rgb_is_imagenet_normalized():
    image = _image()
    out = FeatureBuilder("rgb")(image)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    assert out.shape == (3, 64, 64)
    assert torch.allclose(out, (image - mean) / std)


def test_noise_is_zero_on_constant_image():
    out = FeatureBuilder("noise")(torch.full((3, 32, 32), 0.4))
    assert out.shape == (6, 32, 32)
    assert torch.allclose(out[3:], torch.zeros(3, 32, 32), atol=1e-6)


def test_frequency_constant_image_has_no_ac_energy_and_blocks_are_constant():
    out = FeatureBuilder("frequency")(torch.full((3, 32, 32), 0.7))
    assert torch.allclose(out[3:], torch.zeros(3, 32, 32), atol=1e-6)
    textured = FeatureBuilder("frequency")(_image())[3:]
    assert textured.min() >= 0 and textured.max() <= 1
    # Nearest upsampling: every 8x8 block holds one value per band.
    block = textured[:, :8, :8]
    assert torch.allclose(block, block[:, :1, :1].expand_as(block))


def test_frequency_dct_is_orthonormal():
    builder = FeatureBuilder("frequency")
    assert torch.allclose(builder.dct @ builder.dct.T, torch.eye(8), atol=1e-5)
    assert math.isclose(math.log(65), math.log1p(64))


@pytest.mark.parametrize("architecture", ["unet_resnet34", "segformer_mit_b0"])
@pytest.mark.parametrize("mode", ["rgb", "noise"])
def test_build_model_shapes(architecture, mode):
    model = build_model(architecture, mode, pretrained=False, seed=0).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, input_channels(mode), 64, 64))
    assert logits.shape == (1, 1, 64, 64)


def test_extra_channels_start_at_zero_so_six_channel_model_matches_rgb():
    rgb_model = build_model("unet_resnet34", "rgb", pretrained=False, seed=3).eval()
    six_model = build_model("unet_resnet34", "noise", pretrained=False, seed=3).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a = rgb_model(x)
        b = six_model(torch.cat([x, torch.randn(1, 3, 64, 64)], 1))
    assert torch.allclose(a, b, atol=1e-5)


def test_notebook_style_checkpoint_loads_and_predicts(tmp_path):
    model = build_model("unet_resnet34", "frequency", pretrained=False, seed=1)
    config = {"encoder": "resnet34", "mode": "frequency", "seed": 1, "image_size": [64, 64]}
    path = tmp_path / "best_dice.pt"
    torch.save({"config": config, "epoch": 4, "model": model.state_dict(), "selection_metrics": {}}, path)
    loaded, loaded_config = load_checkpoint(path)
    assert loaded_config["architecture"] == "unet_resnet34"
    (tmp_path / "thresholds.json").write_text(json.dumps({"image_threshold": 0.2}))
    predictor = Predictor(path, thresholds={"image_threshold": 0.2}, device="cpu")
    image = Image.fromarray((np.random.default_rng(0).random((50, 70, 3)) * 255).astype(np.uint8))
    probabilities, mask, summary = predictor.predict(image)
    assert probabilities.shape == (50, 70) and mask.shape == (50, 70)
    assert summary.label in ("Forged", "Authentic")
    assert render_overlay(image, mask, summary).size == (70, 50)


def test_cls_head_model_returns_mask_and_image_logits():
    from forgery_localization.models.factory import split_outputs
    for architecture in ("unet_resnet34", "segformer_mit_b0"):
        model = build_model(architecture, "noise", pretrained=False, seed=0, cls_head=True).eval()
        with torch.no_grad():
            mask_logits, image_logits = split_outputs(model(torch.randn(2, 6, 64, 64)))
        assert mask_logits.shape == (2, 1, 64, 64) and image_logits.shape == (2, 1)
    plain = build_model("unet_resnet34", "rgb", pretrained=False, seed=0).eval()
    with torch.no_grad():
        out, none = split_outputs(plain(torch.randn(1, 3, 64, 64)))
    assert none is None and out.shape == (1, 1, 64, 64)


def test_cls_head_does_not_change_segmentation_init():
    # The head is created after the segmentation layers, so the v2 run starts from the same weights.
    a = build_model("unet_resnet34", "rgb", pretrained=False, seed=5).state_dict()
    b = build_model("unet_resnet34", "rgb", pretrained=False, seed=5, cls_head=True).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(k.startswith("classification_head") for k in b) and set(a) < set(b)


def test_phase2_losses():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "train_multiinput", Path(__file__).resolve().parents[1] / "scripts" / "train_multiinput.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    masks = torch.zeros(2, 1, 16, 16)
    masks[0, :, 4:12, 4:12] = 1                                      # image 0 tampered, image 1 authentic
    logits = torch.randn(2, 1, 16, 16)
    for kind in train.SEG_LOSSES:
        assert torch.isfinite(train.loss_fn(logits, masks, 3.0, kind))
    # Tversky (beta 0.7 > alpha 0.3) separates "missed pixels" from "false alarms" more than Dice does.
    missed = torch.full_like(masks, -8.0)
    missed[0, :, 4:8, 4:12] = 8.0                                    # finds 32 of the 64 forged pixels
    extra = torch.full_like(masks, -8.0)
    extra[0, :, 4:12, 4:12] = 8.0
    extra[0, :, 12:16, 4:12] = 8.0                                   # all 64 + 32 false alarms
    def gap(kind):
        return (train.loss_fn(missed, masks, 3.0, kind) - train.loss_fn(extra, masks, 3.0, kind)).item()
    assert gap("bce_tversky") > gap("bce_dice") > 0
    # The image-level term uses label = mask non-empty and is scaled by cls_weight.
    base = train.loss_fn(logits, masks, 3.0)
    right = torch.tensor([[8.0], [-8.0]])
    wrong = -right
    assert torch.isclose(train.loss_fn(logits, masks, 3.0, cls_logits=right), base, atol=1e-3)
    added = train.loss_fn(logits, masks, 3.0, cls_logits=wrong, cls_weight=0.5) - base
    assert torch.isclose(added, torch.tensor(4.0), atol=1e-2)       # 0.5 x BCE(-8 vs 1) ~= 0.5 x 8


def test_cls_checkpoint_predicts_with_head_score(tmp_path):
    import importlib.util
    from pathlib import Path
    from forgery_localization.models.factory import model_label
    model = build_model("segformer_mit_b0", "rgb", pretrained=False, seed=2, cls_head=True)
    config = {"architecture": "segformer_mit_b0", "mode": "rgb", "seed": 2, "image_size": [64, 64],
              "cls_head": True, "variant": "v2", "protocol": "topic2_v2"}
    path = tmp_path / "best_dice.pt"
    torch.save({"config": config, "epoch": 1, "model": model.state_dict()}, path)
    predictor = Predictor(path, thresholds={"image_threshold": 0.5, "score_method": "cls"}, device="cpu")
    assert predictor.has_cls_head and predictor.name.startswith("segformer_mit_b0-v2 | rgb")
    probabilities, cls = predictor.forward(torch.rand(3, 3, 64, 64))
    assert probabilities.shape == (3, 1, 64, 64) and cls.shape == (3,) and ((cls >= 0) & (cls <= 1)).all()
    image = Image.fromarray((np.random.default_rng(1).random((40, 52, 3)) * 255).astype(np.uint8))
    _, _, summary = predictor.predict(image)
    assert summary.score_method == "cls" and 0 <= summary.image_score <= 1
    assert model_label(config) == "segformer_mit_b0-v2"

    spec = importlib.util.spec_from_file_location(
        "evaluate_script", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    assert evaluate.run_name(config) == "segformer_mit_b0-v2_rgb_seed2"
    assert evaluate.resolve_score_method("auto", True) == "cls"
    assert evaluate.resolve_score_method("auto", False) == "top1pct_mean"
    with pytest.raises(SystemExit):
        evaluate.resolve_score_method("cls", False)
    plain = tmp_path / "plain" / "best_dice.pt"
    plain.parent.mkdir()
    torch.save({"config": {"architecture": "segformer_mit_b0", "mode": "rgb", "seed": 2},
                "model": build_model("segformer_mit_b0", "rgb", pretrained=False).state_dict()}, plain)
    with pytest.raises(ValueError):
        Predictor(plain, thresholds={"score_method": "cls"}, device="cpu")
