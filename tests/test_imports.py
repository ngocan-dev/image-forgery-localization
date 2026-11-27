"""Smoke tests: all scaffold modules must import without accessing data."""

from importlib import import_module

import pytest


@pytest.mark.parametrize("module", [
    "forgery_localization",
    "forgery_localization.datasets.casia_dataset",
    "forgery_localization.models.unet",
    "forgery_localization.training.losses",
    "forgery_localization.training.trainer",
    "forgery_localization.evaluation.metrics",
    "forgery_localization.utils.visualization",
])
def test_module_import(module: str) -> None:
    """Importing a module must not start training or require a dataset."""
    assert import_module(module).__name__ == module
