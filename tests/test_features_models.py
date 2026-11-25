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
