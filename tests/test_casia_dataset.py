"""Synthetic file fixtures only: these are not CASIA ground-truth annotations."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from forgery_localization.datasets.casia_dataset import (
    CASIADataset, split_dataset, validate_split_isolation,
)
from forgery_localization.utils.visualization import visualize_sample


@pytest.fixture
def samples(tmp_path):
    rows = []
    for i in range(12):
        pixels = np.zeros((8, 10, 3), dtype=np.uint8)
        pixels[:, 5:] = (i + 1) * 20
        Image.fromarray(pixels).save(tmp_path / f'image{i}.png')
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:6, 3:8] = 255
        Image.fromarray(mask).save(tmp_path / f'mask{i}.png')
        rows.append(dict(image_path=f'image{i}.png', mask_path=f'mask{i}.png' if i % 2 else '',
                         label='Tp' if i % 2 else 'Au', source_group=f'group{i // 2}',
                         mask_verified='true' if i % 2 else '',
                         mask_source='synthetic test fixture' if i % 2 else ''))
    return tmp_path, rows


def manifest(root, rows, name='samples.csv'):
    path = root / name
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_tensor_contract_interpolation_and_batching(samples):
    root, rows = samples
    dataset = CASIADataset(manifest(root, rows), root=root)
    assert len(dataset) == 12
    for index in (0, 1):
        image, mask = dataset[index]
        assert image.shape == (3, 256, 256)
        assert mask.shape == (1, 256, 256)
        assert image.dtype == mask.dtype == torch.float32
        assert 0 <= image.min() <= image.max() <= 1
        assert set(mask.unique().tolist()) <= {0, 1}
        expected = np.asarray(Image.open(root / rows[index]['image_path']).resize(
            (256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        np.testing.assert_array_equal(image.permute(1, 2, 0).numpy(), expected)
        if index == 0:
            assert mask.sum() == 0
        else:
            expected_mask = np.asarray(Image.open(root / 'mask1.png').resize(
                (256, 256), Image.Resampling.NEAREST)) > 0
            np.testing.assert_array_equal(mask[0].numpy(), expected_mask)
    images, masks = next(iter(DataLoader(dataset, batch_size=2)))
    assert images.shape == (2, 3, 256, 256)
    assert masks.shape == (2, 1, 256, 256)


@pytest.mark.parametrize('kind', ['missing_image', 'missing_mask', 'blank_mask', 'unverified',
    'no_provenance', 'mismatch', 'nonbinary', 'empty', 'colored', 'corrupt',
    'duplicate_image', 'reused_mask', 'bad_label', 'no_group', 'authentic_mask', 'exif'])
def test_invalid_pairs_fail_eagerly(samples, kind):
    root, rows = samples
    if kind == 'missing_image': rows[1]['image_path'] = 'absent.png'
    elif kind == 'missing_mask': rows[1]['mask_path'] = 'absent.png'
    elif kind == 'blank_mask': rows[1]['mask_path'] = ''
    elif kind == 'unverified': rows[1]['mask_verified'] = 'false'
    elif kind == 'no_provenance': rows[1]['mask_source'] = ''
    elif kind == 'mismatch': Image.new('L', (9, 8), 255).save(root / 'mask1.png')
    elif kind == 'nonbinary': Image.new('L', (10, 8), 128).save(root / 'mask1.png')
    elif kind == 'empty': Image.new('L', (10, 8), 0).save(root / 'mask1.png')
    elif kind == 'colored': Image.new('RGB', (10, 8), (255, 0, 0)).save(root / 'mask1.png')
    elif kind == 'corrupt': (root / 'mask1.png').write_text('not an image')
    elif kind == 'duplicate_image': rows[1]['image_path'] = rows[0]['image_path']
    elif kind == 'reused_mask': rows[3]['mask_path'] = rows[1]['mask_path']
    elif kind == 'bad_label': rows[1]['label'] = 'unknown'
    elif kind == 'no_group': rows[1]['source_group'] = ''
    elif kind == 'authentic_mask': rows[0]['mask_path'] = 'mask0.png'
    elif kind == 'exif':
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (10, 8)).save(root / 'image1.png', exif=exif)
    with pytest.raises(ValueError, match='Manifest row'):
        CASIADataset(manifest(root, rows), root)


@pytest.mark.parametrize('mode', ['L', '1', 'RGB', 'P'])
def test_binary_mask_encodings(samples, mode):
    root, rows = samples
    mask = Image.open(root / 'mask1.png').convert(mode)
    mask.save(root / 'mask1.png')
    dataset = CASIADataset(manifest(root, rows), root, image_size=(16, 20))
    image, target = dataset[1]
    assert image.shape == (3, 16, 20)
    assert set(target.unique().tolist()) == {0, 1}
    # Also accept 0/1 grayscale values, not only 0/255.
    Image.fromarray((np.asarray(mask.convert('L')) > 0).astype(np.uint8)).save(root / 'mask1.png')
    assert CASIADataset(manifest(root, rows), root)[1][1].max() == 1


def test_relative_manifest_absolute_images_and_grayscale(samples):
    root, rows = samples
    rows[0]['image_path'] = str((root / 'image0.png').resolve())
    Image.new('L', (10, 8), 70).save(rows[0]['image_path'])
    manifest(root, rows)
    image, _ = CASIADataset('samples.csv', root)[0]
    assert torch.equal(image[0], image[1])


def test_split_reproducibility_and_group_isolation(samples):
    root, rows = samples
    dataset = CASIADataset(manifest(root, rows), root)
    splits = split_dataset(dataset, seed=42)
    assert [s.indices for s in splits] == [s.indices for s in split_dataset(dataset, seed=42)]
    assert [s.indices for s in splits] != [s.indices for s in split_dataset(dataset, seed=1)]
    assert sorted(i for split in splits for i in split.indices) == list(range(12))
    groups = [{dataset.records[i]['source_group'] for i in split.indices} for split in splits]
    assert all(groups)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    separate = [CASIADataset(manifest(root, [rows[i] for i in split.indices], f'{j}.csv'), root)
                for j, split in enumerate(splits)]
    validate_split_isolation(*separate)
    with pytest.raises(ValueError, match='Cross-split'):
        validate_split_isolation(separate[0], separate[0])


def test_duplicate_content_guard(samples):
    root, rows = samples
    (root / 'image2.png').write_bytes((root / 'image0.png').read_bytes())
    with pytest.raises(ValueError, match='Duplicate image content'):
        CASIADataset(manifest(root, rows), root)
    # Across separate manifests, hashes detect duplicates even with different groups.
    first = CASIADataset(manifest(root, rows[:1], 'first.csv'), root)
    second = CASIADataset(manifest(root, rows[2:3], 'second.csv'), root)
    with pytest.raises(ValueError, match='image_hash'):
        validate_split_isolation(first, second)


@pytest.mark.parametrize('ratios', [(1, 0, 0), (0.2, 0.2, 0.2), (float('nan'), .2, .8), (.5, .5)])
def test_invalid_ratios(samples, ratios):
    root, rows = samples
    dataset = CASIADataset(manifest(root, rows), root)
    with pytest.raises(ValueError): split_dataset(dataset, ratios)


def test_too_few_groups_and_bad_manifest(samples):
    root, rows = samples
    dataset = CASIADataset(manifest(root, rows[:2]), root)
    with pytest.raises(ValueError, match='three source groups'): split_dataset(dataset)
    (root / 'bad.csv').write_text('image_path,label\nx,Au\n')
    with pytest.raises(ValueError, match='columns'): CASIADataset('bad.csv', root)
    (root / 'bad.csv').write_text(','.join(rows[0]) + '\n')
    with pytest.raises(ValueError, match='no samples'): CASIADataset('bad.csv', root)


def test_foreground_lost_at_resize_is_not_silent(samples):
    root, rows = samples
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[0, 0] = 255
    Image.fromarray(mask).save(root / 'mask1.png')
    dataset = CASIADataset(manifest(root, rows), root, image_size=(1, 1))
    with pytest.raises(ValueError, match='removed all foreground'): dataset[1]


def test_visualization_overlay_and_save(samples):
    root, rows = samples
    image, mask = CASIADataset(manifest(root, rows), root)[1]
    before_image, before_mask = image.clone(), mask.clone()
    figure = visualize_sample(image, mask)
    assert len(figure.axes) == 3
    overlay = np.asarray(figure.axes[2].images[0].get_array())
    rgb = image.permute(1, 2, 0).numpy()
    foreground = mask[0].numpy().astype(bool)
    np.testing.assert_allclose(overlay[~foreground], rgb[~foreground])
    np.testing.assert_allclose(overlay[foreground], .6 * rgb[foreground] + [.4, 0, 0])
    figure.savefig(root / 'preview.png')
    plt.close(figure)
    assert (root / 'preview.png').stat().st_size > 0
    assert torch.equal(image, before_image) and torch.equal(mask, before_mask)
    with pytest.raises(ValueError): visualize_sample(image, mask, alpha=2)
    with pytest.raises(ValueError): visualize_sample(image, mask + .5)


@pytest.mark.parametrize("image_size", [(0, 256), (256,), (256, 1.5)])
def test_invalid_resize_configuration(samples, image_size):
    root, rows = samples
    with pytest.raises(ValueError, match="image_size"):
        CASIADataset(manifest(root, rows), root, image_size=image_size)


def test_cross_split_reused_mask(samples):
    root, rows = samples
    first = CASIADataset(manifest(root, rows[1:2], "first.csv"), root)
    rows[3]["mask_path"] = rows[1]["mask_path"]
    second = CASIADataset(manifest(root, rows[3:4], "second.csv"), root)
    with pytest.raises(ValueError, match="mask_path"):
        validate_split_isolation(first, second)


def test_mask_removed_after_validation_does_not_become_zero(samples):
    root, rows = samples
    dataset = CASIADataset(manifest(root, rows), root)
    (root / "mask1.png").unlink()
    with pytest.raises(FileNotFoundError):
        dataset[1]
