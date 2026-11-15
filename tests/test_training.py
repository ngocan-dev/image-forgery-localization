"""Loss, metric, trainer and end-to-end tests using synthetic fixtures only."""

from copy import deepcopy
import csv
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from forgery_localization.training.losses import BCEWithLogitsLoss, SoftDiceLoss, CombinedBCEDiceLoss
from forgery_localization.training.trainer import (
    EarlyStopping, resolve_device, train_one_epoch, validate_one_epoch,
)
from forgery_localization.evaluation.metrics import PixelMetrics, pixel_metrics
from forgery_localization.models.unet import UNet

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('train_entrypoint', REPO / 'scripts/train.py')
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


@pytest.fixture(autouse=True)
def small_cpu_tests():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('value', [0., 1.])
@pytest.mark.parametrize('loss_class', [BCEWithLogitsLoss, SoftDiceLoss, CombinedBCEDiceLoss])
def test_losses_empty_full_and_extreme(value, loss_class):
    logits = torch.tensor([-1000., 0., 1000., -1.]).reshape(1, 1, 2, 2).requires_grad_()
    targets = torch.full_like(logits, value)
    loss = loss_class()(logits, targets)
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_loss_values_and_weights():
    logits = torch.zeros(2, 1, 2, 2)
    targets = torch.zeros_like(logits)
    assert BCEWithLogitsLoss()(logits, targets).item() == pytest.approx(math.log(2))
    assert SoftDiceLoss()(logits, targets).item() == pytest.approx(2 / 3)
    expected = 2 * math.log(2) + .5 * (2 / 3)
    assert CombinedBCEDiceLoss(2, .5)(logits, targets).item() == pytest.approx(expected)
    assert SoftDiceLoss()(torch.full_like(logits, -1000), targets).item() == 0
    for ctor, args in [(SoftDiceLoss, (0,)), (CombinedBCEDiceLoss, (0, 0)),
                       (CombinedBCEDiceLoss, (-1, 1)), (SoftDiceLoss, (float('nan'),))]:
        with pytest.raises(ValueError): ctor(*args)


@pytest.mark.parametrize('invalid', ['shape', 'target', 'nan'])
def test_invalid_loss_inputs(invalid):
    logits, targets = torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)
    if invalid == 'shape': targets = targets[0]
    if invalid == 'target': targets.fill_(.2)
    if invalid == 'nan': logits.fill_(float('nan'))
    with pytest.raises(ValueError): CombinedBCEDiceLoss()(logits, targets)


@pytest.mark.parametrize('prediction,target,expected', [
    ([0, 0], [0, 0], [1, 1, 1, 1, 1]),
    ([1, 1], [1, 1], [1, 1, 1, 1, 1]),
    ([0, 0], [1, 1], [0, 0, 0, 0, 0]),
    ([1, 1], [0, 0], [0, 0, 0, 0, 0]),
    ([1, 1, 0, 0], [1, 0, 1, 0], [.5, .5, .5, .5, 1/3]),
])
def test_pixel_metric_cases(prediction, target, expected):
    logits = (torch.tensor(prediction).float() * 20 - 10).reshape(1, 1, 1, -1)
    truth = torch.tensor(target).reshape_as(logits)
    assert list(pixel_metrics(logits, truth).values()) == pytest.approx(expected)


def test_metrics_accumulate_counts_and_threshold():
    metrics = PixelMetrics()
    metrics.update(torch.full((1, 1, 1, 1), 10.), torch.ones(1, 1, 1, 1))
    metrics.update(torch.full((1, 1, 1, 3), 10.), torch.zeros(1, 1, 1, 3))
    assert metrics.compute()['precision'] == .25
    assert metrics.compute()['dice'] == .4
    assert pixel_metrics(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1), .6)['dice'] == 1
    with pytest.raises(ValueError): PixelMetrics(1)


def test_train_validate_and_gradients():
    model = UNet(base_channels=4)
    images = torch.randn(5, 3, 32, 32)
    targets = torch.randint(0, 2, (5, 1, 32, 32)).float()
    loader = DataLoader(TensorDataset(images, targets), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = train_one_epoch(model, loader, optimizer, CombinedBCEDiceLoss())
    assert model.training and all(math.isfinite(x) for x in result.values())
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    optimizer.zero_grad(set_to_none=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = validate_one_epoch(model, loader, CombinedBCEDiceLoss())
    assert not model.training and all(math.isfinite(x) for x in result.values())
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    # Sample-weighted loss, not an unweighted average over 2/2/1 batches.
    with torch.no_grad():
        expected = CombinedBCEDiceLoss()(model(images), targets).item()
    assert result['loss'] == pytest.approx(expected, abs=1e-6)
    empty = DataLoader(TensorDataset(images[:0], targets[:0]))
    with pytest.raises(ValueError, match='empty'): validate_one_epoch(model, empty, CombinedBCEDiceLoss())


def test_nonfinite_loss_and_gradients_rejected():
    model = nn.Conv2d(3, 1, 1)
    loader = [(torch.ones(1, 3, 2, 2), torch.ones(1, 1, 2, 2))]
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(FloatingPointError):
        train_one_epoch(model, loader, optimizer, lambda x, y: x.sum() * float('nan'))
    hook = model.weight.register_hook(lambda gradient: gradient * float('nan'))
    with pytest.raises(FloatingPointError, match='gradient'):
        train_one_epoch(model, loader, optimizer, CombinedBCEDiceLoss())
    hook.remove()


def test_early_stopping_and_device():
    stopper = EarlyStopping(patience=2, mode='min', min_delta=.1)
    assert not stopper.update(1.)
    assert not stopper.update(.95)
    assert stopper.update(.94)
    stopper = EarlyStopping(patience=1)
    assert not stopper.update(.2)
    assert not stopper.update(.3)
    assert stopper.update(.3)
    assert resolve_device('cpu').type == 'cpu'
    assert resolve_device('auto').type in ('cpu', 'cuda')
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match='unavailable'): resolve_device('cuda')


def make_synthetic_config(root):
    """Write tiny synthetic fixtures, explicitly unrelated to real CASIA images."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    for split, count in [('train', 6), ('val', 4)]:
        rows = []
        for index in range(count):
            name = f'{split}_{index}'
            rgb = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(root / f'{name}.png')
            tampered = index % 2 == 1
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 255
            if tampered: Image.fromarray(mask).save(root / f'{name}_mask.png')
            rows.append(dict(image_path=f'{name}.png', mask_path=f'{name}_mask.png' if tampered else '',
                             label='Tp' if tampered else 'Au', source_group=name,
                             mask_verified='true' if tampered else '',
                             mask_source='synthetic unit-test fixture' if tampered else ''))
        with (root / f'{split}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    config = yaml.safe_load((REPO / 'configs/baseline.yaml').read_text())
    config['device'] = 'cpu'
    config['dataset'].update(root=str(root), train_manifest='train.csv', val_manifest='val.csv',
                             test_manifest='DO_NOT_OPEN_TEST.csv', image_size=[32, 32])
    config['model']['base_channels'] = 4  # Same architecture, cheaper smoke-test width.
    config['training'].update(epochs=2, batch_size=2)
    config['paths'].update(checkpoints_dir=str(root / 'checkpoints'), outputs_dir=str(root / 'outputs'))
    return config


def test_entrypoint_resume_matches_uninterrupted(tmp_path):
    config = make_synthetic_config(tmp_path / 'synthetic')
    full_history = entrypoint.run_training(config)
    full = torch.load(Path(config['paths']['checkpoints_dir']) / 'last.pt', weights_only=True)
    partial = deepcopy(config)
    partial['training']['epochs'] = 1
    partial['paths'].update(checkpoints_dir=str(tmp_path / 'resumed'), outputs_dir=str(tmp_path / 'resumed_outputs'))
    entrypoint.run_training(partial)
    last = Path(partial['paths']['checkpoints_dir']) / 'last.pt'
    partial['training']['epochs'] = 2
    history = entrypoint.run_training(partial, resume=last)
    resumed = torch.load(last, weights_only=True)
    assert history == full_history
    for name in full['model_state']:
        torch.testing.assert_close(resumed['model_state'][name], full['model_state'][name], rtol=0, atol=0)
    best = torch.load(last.with_name('best.pt'), weights_only=True)
    assert best['val_dice'] == max(row['val_dice'] for row in history)
    model = UNet(base_channels=4)
    model.load_state_dict(best['model_state'])
    assert torch.isfinite(model(torch.zeros(1, 3, 32, 32))).all()
    with (Path(partial['paths']['outputs_dir']) / 'history.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
    with pytest.raises(FileExistsError): entrypoint.run_training(partial)
    with pytest.raises(ValueError, match='last.pt'): entrypoint.run_training(partial, last.with_name('best.pt'))
    partial['training']['learning_rate'] = .1
    with pytest.raises(ValueError, match='configuration mismatch'): entrypoint.run_training(partial, last)
    partial['training']['learning_rate'] = config['training']['learning_rate']
    mask_path = Path(config['dataset']['root']) / 'train_1_mask.png'
    Image.new('L', (32, 32), 255).save(mask_path)
    with pytest.raises(ValueError, match='Dataset changed'): entrypoint.run_training(partial, last)


def test_cli_and_early_stop(tmp_path):
    config = make_synthetic_config(tmp_path / 'cli')
    config['training']['epochs'] = 5
    # A delta larger than possible Dice improvements forces stopping at epoch 2.
    config['training']['early_stopping'].update(patience=1, min_delta=2.)
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    env = dict(os.environ, PYTHONPATH=str(REPO / 'src'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    result = subprocess.run([sys.executable, str(REPO / 'scripts/train.py'), '--config', str(config_path)],
                            env=env, cwd=REPO, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert 'Early stopping after epoch 2' in result.stdout
    checkpoint = torch.load(Path(config['paths']['checkpoints_dir']) / 'last.pt', weights_only=True)
    assert checkpoint['epoch'] == 2
    assert not (Path(config['dataset']['root']) / 'DO_NOT_OPEN_TEST.csv').exists()
    assert len(entrypoint.run_training(config, Path(config['paths']['checkpoints_dir']) / 'last.pt')) == 2


def test_leakage_rejected_before_training(tmp_path):
    config = make_synthetic_config(tmp_path / 'leak')
    config['dataset']['val_manifest'] = 'train.csv'
    with pytest.raises(ValueError, match='Cross-split'): entrypoint.run_training(config)
    assert not Path(config['paths']['checkpoints_dir']).exists()
