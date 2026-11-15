"""Kaggle environment setup and explicitly selected smoke/full training stages."""
import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

REPO = Path(__file__).resolve().parents[1]


def setup(allow_runtime_python=False):
    if sys.version_info[:2] != (3, 11) and not allow_runtime_python:
        raise RuntimeError('Project metadata targets Python 3.11. To test the existing Kaggle runtime explicitly, pass --allow-runtime-python.')
    if sys.version_info < (3, 11):
        raise RuntimeError('Python 3.11 or newer is required for this setup script')
    from packaging.requirements import Requirement
    metadata = tomllib.loads((REPO / 'pyproject.toml').read_text())['project']
    torch_requirement = next(Requirement(dep) for dep in metadata['dependencies'] if Requirement(dep).name == 'torch')
    import torch
    if not torch_requirement.specifier.contains(torch.__version__):
        raise RuntimeError(f'Existing PyTorch {torch.__version__} does not satisfy {torch_requirement}; choose a compatible Kaggle runtime')
    # Keep Kaggle's preinstalled GPU PyTorch; do not replace it with a CPU wheel.
    requirements = [dep for dep in metadata['dependencies'] if Requirement(dep).name != 'torch']
    subprocess.run([sys.executable, '-m', 'pip', 'install', *requirements, *metadata['optional-dependencies']['dev']], check=True)
    command = [sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', str(REPO)]
    if sys.version_info[:2] != (3, 11): command.append('--ignore-requires-python')
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider'], cwd=REPO, check=True)
    environment = dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
                       cuda_available=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       python_311_target_met=sys.version_info[:2] == (3, 11), tests_passed=True,
                       dependencies={Requirement(dep).name: importlib.metadata.version(Requirement(dep).name)
                                     for dep in metadata['dependencies']})
    target = Path('/kaggle/working/environment.json')
    target.write_text(json.dumps(environment, indent=2))
    print(json.dumps(environment, indent=2))


def train(prepared, stage, resume=None, confirm_smoke_reviewed=False):
    import torch
    import yaml
    prepared = Path(prepared).resolve()
    working = Path('/kaggle/working').resolve()
    if working not in prepared.parents:
        raise ValueError('Prepared artifacts must be under /kaggle/working/')
    if not (prepared / 'validated.json').is_file():
        raise ValueError('Run audit finalize successfully before training')
    validation = json.loads((prepared / 'validated.json').read_text())
    for path, digest in validation['file_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Validated file changed: {path}; re-audit before training')
    config_path = prepared / f'{stage}.yaml'
    config = yaml.safe_load(config_path.read_text())
    if Path(config['dataset']['root']).resolve() != prepared:
        raise ValueError('Configuration must use the validated dataset root')
    for split in ('train', 'val'):
        expected = f'{"smoke_" if stage == "smoke" else ""}{split}.csv'
        if config['dataset'][f'{split}_manifest'] != expected:
            raise ValueError('Configuration must use the validated manifests')
    for path in config['paths'].values():
        if working not in Path(path).resolve().parents:
            raise ValueError('All training outputs must remain under /kaggle/working/')
    if stage == 'full':
        if not confirm_smoke_reviewed:
            raise ValueError('Inspect the real-data smoke log/overlays, then pass --confirm-smoke-reviewed')
        if not torch.cuda.is_available(): raise RuntimeError('Enable a Kaggle GPU accelerator before full training')
        smoke = torch.load(prepared / 'smoke/checkpoints/last.pt', map_location='cpu', weights_only=True)
        if smoke.get('epoch', 0) < 1: raise ValueError('Complete the real-data smoke test first')
    command = [sys.executable, str(REPO / 'scripts/train.py'), '--config', str(config_path)]
    if resume: command.extend(['--resume', str(resume)])
    env = dict(os.environ, PYTHONPATH=str(REPO / 'src'))
    subprocess.run(command, cwd=REPO, env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['setup', 'smoke', 'full'])
    parser.add_argument('--allow-runtime-python', action='store_true')
    parser.add_argument('--prepared', type=Path, default=Path('/kaggle/working/casia_prepared'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--confirm-smoke-reviewed', action='store_true')
    args = parser.parse_args(argv)
    if args.stage == 'setup': setup(args.allow_runtime_python)
    else: train(args.prepared, args.stage, args.resume, args.confirm_smoke_reviewed)


if __name__ == '__main__':
    main()
