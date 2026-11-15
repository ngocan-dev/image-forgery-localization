"""Kaggle audit workflow tests; all images here are synthetic fixtures."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from forgery_localization.datasets.casia_dataset import CASIADataset

REPO = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module('audit_dataset')
kaggle = module('kaggle_setup')


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / 'CASIA2'
    for name in ('Au', 'Tp', 'Groundtruth'): (root / name).mkdir(parents=True)
    rng = np.random.default_rng(12)
    for i in range(6):
        for label in ('Au', 'Tp'):
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root / label / f'{label}_{i}.png')
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        if i == 1: mask = 255 - mask
        Image.fromarray(mask).save(root / 'Groundtruth' / f'Tp_{i}_gt.png')
    return root, tmp_path / 'audit', tmp_path / 'prepared'


def review(directory):
    with (directory / 'review.csv').open(newline='') as stream: rows = list(csv.DictReader(stream))
    for row in rows:
        identifier = Path(row['image_path']).stem.split('_')[-1]
        row.update(action='include', source_group=f'source_{identifier}')
        if row['label'] == 'Tp':
            row.update(alignment_verified='true', polarity='black' if identifier == '1' else 'white',
                       mask_source='synthetic test fixture, manually specified test polarity')
    audit.write_csv(directory / 'review.csv', rows, audit.REVIEW)
    return rows


def test_audit_review_export(fixture):
    root, output, destination = fixture
    report = audit.audit(root, output, preview_limit=1)
    assert report['counts'] == {'Au': 6, 'Tp': 6, 'Groundtruth': 6}
    assert not report['polarity_verified']
    assert len(list((output / 'previews').glob('*.png'))) == 1
    with pytest.raises(ValueError, match='Every row'): audit.finalize(output, destination)
    review(output)
    result = audit.finalize(output, destination)
    assert sum(sum(counts.values()) for counts in result['counts'].values()) == 12
    dataset = CASIADataset(destination / 'all.csv')
    for i, row in enumerate(dataset.records):
        if row['label'] == 'tampered': assert dataset[i][1].sum() == 128 * 128
    full = yaml.safe_load((destination / 'full.yaml').read_text())
    smoke = yaml.safe_load((destination / 'smoke.yaml').read_text())
    assert full['device'] == 'cuda' and full['training']['epochs'] == 30
    assert smoke['training']['epochs'] == 1 and smoke['training']['batch_size'] == 2
    for mode in (full, smoke):
        assert str(destination) in mode['paths']['checkpoints_dir']
    repeated = destination.with_name('prepared_again')
    audit.finalize(output, repeated)
    # Canonical source images split identically even when normalized-mask paths differ.
    for split in ('train', 'val', 'test'):
        a = CASIADataset(destination / f'{split}.csv')
        b = CASIADataset(repeated / f'{split}.csv')
        assert [r['image_path'] for r in a.records] == [r['image_path'] for r in b.records]


@pytest.mark.parametrize('kind, expected', [('missing', 'missing_mask'), ('mismatch', 'dimension_mismatch'),
                                          ('invalid', 'not binary'), ('ambiguous', 'ambiguous_masks')])
def test_pairing_issues_reported(fixture, kind, expected):
    root, output, destination = fixture
    mask = root / 'Groundtruth/Tp_0_gt.png'
    if kind == 'missing': mask.unlink()
    elif kind == 'mismatch': Image.new('L', (4, 4), 255).save(mask)
    elif kind == 'invalid': Image.new('L', (32, 32), 128).save(mask)
    elif kind == 'ambiguous': Image.open(mask).save(mask.with_suffix('.bmp'))
    report = audit.audit(root, output, preview_limit=0)
    assert any(expected in row['status'] for row in report['rows'])
    rows = review(output)
    with pytest.raises((ValueError, FileNotFoundError)): audit.finalize(output, destination)
    for row in rows:
        if Path(row['image_path']).name == 'Tp_0.png': row.update(action='exclude', reason='synthetic invalid fixture')
    audit.write_csv(output / 'review.csv', rows, audit.REVIEW)
    result = audit.finalize(output, destination)
    assert result['excluded'] == 1


def test_changed_input_and_unreviewed_polarity_rejected(fixture):
    root, output, destination = fixture
    audit.audit(root, output, preview_limit=0)
    rows = review(output)
    rows[-1]['polarity'] = ''
    audit.write_csv(output / 'review.csv', rows, audit.REVIEW)
    with pytest.raises(ValueError, match='polarity'): audit.finalize(output, destination)
    review(output)
    Image.new('RGB', (32, 32)).save(root / 'Au/Au_0.png')
    with pytest.raises(ValueError, match='changed'): audit.finalize(output, destination)


def test_no_missing_rows_or_input_overwrite(fixture):
    root, output, destination = fixture
    with pytest.raises(ValueError): audit.audit(root, root / 'audit')
    audit.audit(root, output, preview_limit=0)
    with pytest.raises(FileExistsError): audit.audit(root, output)
    rows = review(output)
    audit.write_csv(output / 'review.csv', rows[:-1], audit.REVIEW)
    with pytest.raises(ValueError, match='exactly'): audit.finalize(output, destination)


def test_kaggle_path_and_runtime_guards(tmp_path):
    with pytest.raises(ValueError, match='/kaggle/working'): kaggle.train(tmp_path, 'smoke')
    import sys
    if sys.version_info[:2] != (3, 11):
        with pytest.raises(RuntimeError, match='Python 3.11'): kaggle.setup()


def test_exported_smoke_config_trains_synthetic_only(fixture):
    import torch
    root, output, destination = fixture
    audit.audit(root, output, preview_limit=0)
    review(output)
    audit.finalize(output, destination)
    test_hash = audit.sha(destination / 'test.csv')
    config = yaml.safe_load((destination / 'smoke.yaml').read_text())
    config['dataset']['image_size'] = [32, 32]
    config['model']['base_channels'] = 4
    config['device'] = 'cpu'
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        history = module('train').run_training(config)
    finally:
        torch.set_num_threads(previous)
    assert len(history) == 1
    saved = torch.load(destination / 'smoke/checkpoints/last.pt', weights_only=True)
    assert saved['epoch'] == 1
    assert audit.sha(destination / 'test.csv') == test_hash


def test_export_never_writes_input(fixture):
    root, output, destination = fixture
    audit.audit(root, output, preview_limit=0)
    review(output)
    with pytest.raises(ValueError, match='outside'): audit.finalize(output, root / 'export')
