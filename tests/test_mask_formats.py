"""Format and approval regressions using synthetic masks, not real annotations."""
import numpy as np
from PIL import Image
import pytest
import torch

from forgery_localization.datasets.mask_utils import read_mask_values, binarize_mask, threshold_options
from forgery_localization.datasets.casia_dataset import CASIADataset
from tests.test_casia_dataset import samples, manifest
from tests.test_kaggle_workflow import fixture, review, audit


def save_mask(path, gray, mode):
    if mode == 'L': array = gray
    else:
        array = np.repeat(gray[..., None], 3, axis=2)
        if mode == 'RGBA': array = np.concatenate([array, np.full((*gray.shape, 1), 255, np.uint8)], axis=2)
    Image.fromarray(array).save(path)


@pytest.mark.parametrize('mode', ['L', 'RGB', 'RGBA'])
@pytest.mark.parametrize('foreground', [1, 255])
def test_binary_formats(samples, mode, foreground):
    root, rows = samples
    gray = np.zeros((8, 10), np.uint8)
    gray[2:6, 3:8] = foreground
    save_mask(root / 'mask1.png', gray, mode)
    ds = CASIADataset(manifest(root, rows), root, image_size=(8, 10))
    mask = ds[1][1]
    assert mask.shape == (1, 8, 10) and mask.dtype == torch.float32
    np.testing.assert_array_equal(mask[0], gray > 0)
    np.testing.assert_array_equal(audit.binary_mask(root / 'mask1.png'), gray > 0)


@pytest.mark.parametrize('mode', ['L', 'RGB', 'RGBA'])
def test_explicit_threshold_boundary_and_shared_decoding(samples, mode):
    root, rows = samples
    gray = np.tile(np.array([0, 1, 126, 127, 128, 254, 255, 0, 0, 0], np.uint8), (8, 1))
    save_mask(root / 'mask1.png', gray, mode)
    with pytest.raises(ValueError, match='approval'): CASIADataset(manifest(root, rows), root)
    for row in rows: row.update(threshold_approved='false', mask_threshold='127')
    with pytest.raises(ValueError, match='approval'): CASIADataset(manifest(root, rows), root)
    rows[1]['threshold_approved'] = 'true'
    ds = CASIADataset(manifest(root, rows), root, image_size=(8, 10))
    np.testing.assert_array_equal(ds[1][1][0], gray > 127)
    np.testing.assert_array_equal(audit.binary_mask(root / 'mask1.png', threshold_approved=True), gray > 127)
    rows[1]['mask_threshold'] = '128'
    ds = CASIADataset(manifest(root, rows), root, image_size=(8, 10))
    np.testing.assert_array_equal(ds[1][1][0], gray > 128)
    rows[1]['mask_verified'] = 'false'
    with pytest.raises(ValueError, match='mask_verified'): CASIADataset(manifest(root, rows), root)


@pytest.mark.parametrize('alpha', [0, 1, 128, 254])
def test_nonopaque_alpha_never_thresholded(samples, alpha):
    root, rows = samples
    array = np.full((8, 10, 4), 255, np.uint8)
    array[0, 0, 3] = alpha
    Image.fromarray(array).save(root / 'mask1.png')
    for row in rows: row.update(threshold_approved='true')
    with pytest.raises(ValueError, match='alpha'): CASIADataset(manifest(root, rows), root)
    with pytest.raises(ValueError, match='alpha'): audit.binary_mask(root / 'mask1.png', threshold_approved=True)


@pytest.mark.parametrize('mode', ['RGB', 'RGBA'])
def test_nonidentical_channels_rejected_even_with_approval(samples, mode):
    root, rows = samples
    array = np.full((8, 10, 3 if mode == 'RGB' else 4), 255, np.uint8)
    array[0, 0, 1] = 254
    Image.fromarray(array).save(root / 'mask1.png')
    for row in rows: row.update(threshold_approved='true')
    with pytest.raises(ValueError, match='identical'): CASIADataset(manifest(root, rows), root)
    with pytest.raises(ValueError, match='identical'): audit.binary_mask(root / 'mask1.png', threshold_approved=True)


@pytest.mark.parametrize('row', [{'threshold_approved': 'yes'}, {'mask_threshold': '127.5'},
                                {'mask_threshold': '-1'}, {'mask_threshold': '255'}])
def test_invalid_options(row):
    with pytest.raises(ValueError): threshold_options(row)


def test_threshold_does_not_break_binary_one_encoding():
    np.testing.assert_array_equal(binarize_mask(np.array([[0, 1]], np.uint8), threshold_approved=True), [[False, True]])


def test_threshold_cannot_approve_size_or_empty_foreground(samples):
    root, rows = samples
    for row in rows: row.update(threshold_approved='true')
    Image.new('RGBA', (9, 8), (128, 128, 128, 255)).save(root / 'mask1.png')
    with pytest.raises(ValueError, match='dimensions'): CASIADataset(manifest(root, rows), root)
    Image.new('L', (10, 8), 127).save(root / 'mask1.png')
    with pytest.raises(ValueError, match='no foreground'): CASIADataset(manifest(root, rows), root)


def test_audit_requires_per_mask_threshold_and_polarity_review(fixture):
    root, output, destination = fixture
    path = root / 'Groundtruth/Tp_0_gt.png'
    gray = np.zeros((32, 32), np.uint8)
    gray[8:24, 8:24] = 255
    gray[0, 0], gray[0, 1] = 127, 128
    save_mask(path, gray, 'RGBA')
    report = audit.audit(root, output, preview_limit=0)
    row = next(r for r in report['rows'] if r['mask_path'] == str(path))
    assert row['intermediate_pixels'] == 2
    assert 'approval' in row['status']
    rows = review(output)
    with pytest.raises(ValueError, match='approval'): audit.finalize(output, destination)
    selected = next(r for r in rows if r['mask_path'] == str(path))
    selected.update(threshold_approved='true', mask_threshold='127', polarity='')
    audit.write_csv(output / 'review.csv', rows, audit.REVIEW)
    with pytest.raises(ValueError, match='polarity'): audit.finalize(output, destination)
    selected['polarity'] = 'black'
    audit.write_csv(output / 'review.csv', rows, audit.REVIEW)
    audit.finalize(output, destination)
    dataset = CASIADataset(destination / 'all.csv', image_size=(32, 32))
    index = next(i for i, r in enumerate(dataset.records) if r['image_path'].name == 'Tp_0.png')
    np.testing.assert_array_equal(dataset[index][1][0], ~(gray > 127))
    assert 'threshold_approved=True; threshold=127' in dataset.records[index]['mask_source']
