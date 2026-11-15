"""Two-stage CASIA2 audit: inventory first, explicit human review before export."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import yaml

from forgery_localization.datasets.casia_dataset import CASIADataset, split_dataset, validate_split_isolation

FIELDS = ['image_path', 'mask_path', 'label', 'source_group', 'mask_verified', 'mask_source']
REVIEW = ['image_path', 'mask_path', 'label', 'image_sha256', 'mask_sha256', 'status',
          'white_fraction', 'action', 'reason', 'source_group', 'alignment_verified', 'polarity', 'mask_source']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def binary_mask(path):
    with Image.open(path) as source:
        if source.getexif().get(274, 1) != 1:
            raise ValueError('mask EXIF rotation requires paired preprocessing')
        if source.mode == 'P':
            if 'transparency' in source.info: raise ValueError('transparent mask')
            source = source.convert('RGB')
        array = np.array(source)
    if array.ndim == 3:
        if array.shape[2] != 3 or not np.all(array == array[..., :1]):
            raise ValueError('colored/alpha mask')
        array = array[..., 0]
    values = set(np.unique(array).tolist())
    if array.ndim != 2 or not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError('mask is not binary 0/1 or 0/255')
    return array > 0


def read_rgb(path):
    with Image.open(path) as source:
        if source.getexif().get(274, 1) != 1:
            raise ValueError('image EXIF rotation requires paired preprocessing')
        return source.convert('RGB')


def preview(image_path, mask_path, destination):
    """Show both interpretations; intensity alone cannot establish polarity."""
    rgb = read_rgb(image_path).resize((256, 256), Image.Resampling.BILINEAR)
    mask = Image.fromarray(binary_mask(mask_path)).resize((256, 256), Image.Resampling.NEAREST)
    array = np.array(rgb)
    foreground = np.array(mask)
    canvas = Image.new('RGB', (1024, 280), 'white')
    panels = [rgb, mask.convert('RGB')]
    for selected in (foreground, ~foreground):
        overlay = array.copy()
        overlay[selected] = .6 * overlay[selected] + .4 * np.array([255, 0, 0])
        panels.append(Image.fromarray(overlay))
    draw = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(zip(panels, ['RGB', 'Mask', 'White = manipulated', 'Black = manipulated'])):
        canvas.paste(panel, (256 * i, 24))
        draw.text((256 * i + 4, 4), title, fill='black')
    canvas.save(destination)


def audit(root, output, suffix='_gt', preview_limit=24):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents:
        raise ValueError('Audit output must be outside the input dataset')
    folders = {name: root / name for name in ('Au', 'Tp', 'Groundtruth')}
    for folder in folders.values():
        if not folder.is_dir(): raise FileNotFoundError(f'Missing directory: {folder}')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new audit directory to preserve previous review')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'previews').mkdir()
    files = {name: sorted(p for p in folder.rglob('*') if p.is_file()) for name, folder in folders.items()}
    index = {}
    for path in files['Groundtruth']:
        key = path.stem.casefold()
        if suffix and key.endswith(suffix.casefold()): key = key[:-len(suffix)]
        index.setdefault(key, []).append(path)
    rows, used, preview_count = [], set(), 0
    for label, folder in [('Au', 'Au'), ('Tp', 'Tp')]:
        for image_path in files[folder]:
            candidates = index.get(image_path.stem.casefold(), []) if label == 'Tp' else []
            mask_path = candidates[0] if len(candidates) == 1 else None
            row = dict.fromkeys(REVIEW, '')
            row.update(image_path=str(image_path), label=label, image_sha256=sha(image_path), action='review')
            problems = []
            if label == 'Tp' and len(candidates) != 1:
                problems.append('missing_mask' if not candidates else 'ambiguous_masks')
            if mask_path:
                used.add(mask_path)
                row.update(mask_path=str(mask_path), mask_sha256=sha(mask_path))
            try:
                rgb = read_rgb(image_path)
                if mask_path:
                    mask = binary_mask(mask_path)
                    row['white_fraction'] = float(mask.mean())
                    if mask.shape != (rgb.height, rgb.width): raise ValueError('dimension_mismatch')
                    if mask.all() or not mask.any(): problems.append('constant_mask_requires_review')
                    # Preview is diagnostic only, never an automatic approval.
                    if preview_count < preview_limit:
                        preview(image_path, mask_path, output / 'previews' / f'{len(rows):06d}.png')
                        preview_count += 1
            except (OSError, ValueError) as error:
                problems.append(str(error))
            row['status'] = '; '.join(problems) or 'structural_checks_passed_review_required'
            rows.append(row)
    write_csv(output / 'review.csv', rows, REVIEW)
    # Immutable inventory is compared with the edited review before exporting.
    report = dict(root=str(root), counts={k: len(v) for k, v in files.items()},
                  expected_counts_user_supplied={'Au': 7492, 'Tp': 5124, 'Groundtruth': 5123},
                  rows=rows, orphan_masks=[str(p) for p in files['Groundtruth'] if p not in used],
                  mask_suffix=suffix, polarity_verified=False, semantic_alignment_verified=False)
    (output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    return report


def finalize(audit_dir, destination, seed=42):
    audit_dir, destination = Path(audit_dir).resolve(), Path(destination).resolve()
    report = json.loads((audit_dir / 'audit.json').read_text())
    input_root = Path(report['root']).resolve()
    if destination == input_root or input_root in destination.parents:
        raise ValueError('Export output must be outside the input dataset')
    if destination.exists() and any(destination.iterdir()): raise FileExistsError('Use an empty export directory')
    with (audit_dir / 'review.csv').open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    inventory = {r['image_path']: r for r in report['rows']}
    if len(rows) != len(inventory) or {r['image_path'] for r in rows} != set(inventory):
        raise ValueError('Review must contain exactly the audited image rows, including exclusions')
    accepted, excluded = [], []
    for row in rows:
        original = inventory[row['image_path']]
        for key in ('image_path', 'mask_path', 'label', 'image_sha256', 'mask_sha256'):
            if row[key] != original[key]: raise ValueError(f'Do not edit audited pairing fields: {key}; correct input and re-audit')
        if row['action'] == 'exclude':
            if not row['reason'].strip(): raise ValueError('Exclusions require a reason')
            excluded.append(row)
            continue
        if row['action'] != 'include' or not row['source_group'].strip():
            raise ValueError('Every row needs include + reviewed source_group, or exclude + reason')
        if sha(row['image_path']) != row['image_sha256']: raise ValueError('Image changed after audit')
        rgb = read_rgb(row['image_path'])
        if row['label'] == 'Tp':
            if not row['mask_path'] or sha(row['mask_path']) != row['mask_sha256']:
                raise ValueError('Missing or changed mask')
            if row['alignment_verified'].lower() != 'true' or row['polarity'] not in ('white', 'black') or not row['mask_source'].strip():
                raise ValueError('Tampered rows require alignment_verified=true, white/black polarity and mask_source')
            mask = binary_mask(row['mask_path'])
            if mask.shape != (rgb.height, rgb.width): raise ValueError('Mask/image dimensions mismatch')
            if row['polarity'] == 'black': mask = ~mask
            if not mask.any(): raise ValueError('Reviewed tampered mask has no foreground')
        accepted.append(row)
    if not accepted: raise ValueError('No reviewed samples accepted')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'masks').mkdir()
    records = []
    for i, row in enumerate(accepted):
        target = ''
        if row['label'] == 'Tp':
            mask = binary_mask(row['mask_path'])
            if row['polarity'] == 'black': mask = ~mask
            target = str(destination / 'masks' / f'{i:06d}.png')
            Image.fromarray(mask.astype(np.uint8) * 255).save(target)
        records.append(dict(image_path=row['image_path'], mask_path=target, label=row['label'],
                            source_group=row['source_group'].strip(), mask_verified='true' if target else '',
                            mask_source=row['mask_source'] + f"; original={row['mask_path']}; sha256={row['mask_sha256']}; foreground={row['polarity']}" if target else ''))
    write_csv(destination / 'all.csv', records, FIELDS)
    dataset = CASIADataset(destination / 'all.csv')
    # Check resizing too, including tiny foreground that disappears at 256x256.
    for i in range(len(dataset)): dataset[i]
    splits = split_dataset(dataset, seed=seed)
    counts = {}
    for name, subset in zip(('train', 'val', 'test'), splits):
        selected = [records[i] for i in subset.indices]
        write_csv(destination / f'{name}.csv', selected, FIELDS)
        counts[name] = {label: sum(r['label'] == label for r in selected) for label in ('Au', 'Tp')}
        # Small real-data smoke subsets come ONLY from train/val, never test.
        if name != 'test':
            small = []
            for label in ('Au', 'Tp'): small.extend([r for r in selected if r['label'] == label][:4])
            write_csv(destination / f'smoke_{name}.csv', small, FIELDS)
    validate_split_isolation(*(CASIADataset(destination / f'{name}.csv') for name in ('train', 'val', 'test')))
    baseline = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/baseline.yaml').read_text())
    for mode in ('smoke', 'full'):
        config = json.loads(json.dumps(baseline))
        config['dataset'].update(root=str(destination), train_manifest=f'{"smoke_" if mode == "smoke" else ""}train.csv',
                                 val_manifest=f'{"smoke_" if mode == "smoke" else ""}val.csv', test_manifest='test.csv')
        config['seed'] = seed
        config['device'] = 'auto' if mode == 'smoke' else 'cuda'
        config['training']['epochs'] = 1 if mode == 'smoke' else 30
        config['training']['batch_size'] = 2 if mode == 'smoke' else 8
        for key, folder in [('checkpoints_dir', 'checkpoints'), ('outputs_dir', 'outputs'), ('reports_dir', 'reports')]:
            config['paths'][key] = str(destination / mode / folder)
        (destination / f'{mode}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_csv(destination / 'exclusions.csv', excluded, REVIEW)
    result = dict(seed=seed, counts=counts, excluded=len(excluded), source_group_review='user supplied; unknown relationships cannot be detected',
                  review_sha256=sha(audit_dir / 'review.csv'), provenance_audit=str(audit_dir / 'audit.json'))
    bound_paths = {r['image_path'] for r in records} | {r['mask_path'] for r in records if r['mask_path']}
    bound_paths.update(str(p) for p in destination.glob('*.csv'))
    result['file_sha256'] = {p: sha(p) for p in sorted(bound_paths)}
    (destination / 'validated.json').write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    scan = sub.add_parser('scan')
    scan.add_argument('--root', required=True, type=Path)
    scan.add_argument('--output', required=True, type=Path)
    scan.add_argument('--mask-suffix', default='_gt', help='Optional mask stem suffix; no fuzzy matching')
    scan.add_argument('--preview-limit', type=int, default=24)
    export = sub.add_parser('finalize')
    export.add_argument('--audit-dir', required=True, type=Path)
    export.add_argument('--output', required=True, type=Path)
    export.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == 'scan': return audit(args.root, args.output, args.mask_suffix, args.preview_limit)
    return finalize(args.audit_dir, args.output, args.seed)


if __name__ == '__main__':
    main()
