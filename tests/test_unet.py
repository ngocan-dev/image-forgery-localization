"""Synthetic model checks; no CASIA data or training pipeline is required."""

import pytest
import torch
from torch import nn

from forgery_localization.models.unet import UNet


@pytest.fixture(autouse=True)
def reproducible_cpu_tests():
    """Keep CPU checks quick and isolate their random-number state."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            yield
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rgb_forward_shape(batch_size):
    model = UNet().eval()
    with torch.no_grad():
        logits = model(torch.randn(batch_size, 3, 256, 256))
    assert logits.shape == (batch_size, 1, 256, 256)
    assert torch.isfinite(logits).all()


def test_backward_reaches_every_stage():
    model = UNet()
    images = torch.randn(1, 3, 256, 256, requires_grad=True)
    targets = torch.randint(0, 2, (1, 1, 256, 256)).float()
    loss = nn.BCEWithLogitsLoss()(model(images), targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_output_is_raw_logits():
    model = UNet(base_channels=4).eval()
    # Force a known negative logit: a sigmoid output could never equal -2.
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(-2.0)
        logits = model(torch.zeros(1, 3, 16, 16))
    torch.testing.assert_close(logits, torch.full_like(logits, -2.0))


@pytest.mark.parametrize("shape", [(1, 3, 16, 16), (1, 3, 33, 49)])
def test_small_and_odd_spatial_sizes(shape):
    with torch.no_grad():
        logits = UNet(base_channels=4)(torch.randn(shape))
    assert logits.shape == (shape[0], 1, *shape[-2:])


def test_configurable_channels():
    with torch.no_grad():
        logits = UNet(in_channels=5, out_channels=2, base_channels=4)(
            torch.randn(1, 5, 32, 32)
        )
    assert logits.shape == (1, 2, 32, 32)


@pytest.mark.parametrize("shape, message", [
    ((3, 32, 32), "Expected input shape"),
    ((1, 1, 32, 32), "Expected 3 input channels"),
    ((1, 3, 15, 32), "at least 16"),
    ((1, 3, 32, 15), "at least 16"),
])
def test_invalid_input(shape, message):
    with pytest.raises(ValueError, match=message):
        UNet(base_channels=4)(torch.randn(shape))


@pytest.mark.parametrize("kwargs", [
    {"in_channels": 0}, {"out_channels": -1}, {"base_channels": 0},
    {"base_channels": 1.5}, {"base_channels": True},
])
def test_invalid_channel_configuration(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        UNet(**kwargs)
