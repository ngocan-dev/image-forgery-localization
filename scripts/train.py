"""Train the existing RGB U-Net using verified train/validation manifests.

Run from the repository root after installing the package (pip install -e .).
The test manifest is never opened by this command.
"""

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from forgery_localization.datasets.casia_dataset import CASIADataset, validate_split_isolation
from forgery_localization.models.unet import UNet
from forgery_localization.training.losses import CombinedBCEDiceLoss
from forgery_localization.training.trainer import (
    EarlyStopping, resolve_device, train_one_epoch, validate_one_epoch,
)


def _atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_history(history, path):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def _fingerprint(dataset):
    """Bind resume to sample order, labels, groups and image/mask content."""
    digest = hashlib.sha256()
    for record in dataset.records:
        digest.update(json.dumps({k: str(v) for k, v in record.items()}, sort_keys=True).encode())
        if record["mask_path"] is not None:
            digest.update(record["mask_path"].read_bytes())
    return digest.hexdigest()


def _resume_settings(config):
    # Allow extending the total epoch budget, changing device or output location.
    settings = deepcopy(config)
    settings.pop("paths", None)
    settings.pop("device", None)
    settings["training"].pop("epochs", None)
    return settings


def run_training(config, resume=None):
    """Return history after training; all paths follow the repository-root convention."""
    config = deepcopy(config)
    training, data, model_config = config["training"], config["dataset"], config["model"]
    if model_config.get("name") != "unet" or (model_config["in_channels"], model_config["out_channels"]) != (3, 1):
        raise ValueError("This pipeline requires RGB U-Net with 3 input and 1 output channels")
    if training["optimizer"].lower() != "adamw":
        raise ValueError("Only AdamW is implemented")
    if training["loss"]["name"] != "bce_with_logits_plus_dice":
        raise ValueError("Only bce_with_logits_plus_dice is implemented")
    for key in ("batch_size", "epochs"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(training.get("num_workers", 0)) is not int or training.get("num_workers", 0) < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    for key in ("learning_rate", "weight_decay"):
        value = training[key]
        if not math.isfinite(value) or value < 0 or (key == "learning_rate" and value == 0):
            raise ValueError(f"Invalid {key}")
    threshold = config["evaluation"]["threshold"]
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")
    early_config = training.get("early_stopping", {})
    monitor = early_config.get("monitor", "dice")
    if monitor not in ("loss", "dice"):
        raise ValueError("Early stopping monitor must be loss or dice")
    stopper = EarlyStopping(early_config.get("patience", 10),
                            "min" if monitor == "loss" else "max",
                            early_config.get("min_delta", 0.0))
    device = resolve_device(config.get("device", "auto"))
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic ordering and cuDNN settings; cross-platform bit identity is
    # not promised, especially across PyTorch versions or CPU/CUDA resumes.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(seed)

    # The SAME loader and preprocessing settings are used for both splits.
    datasets = [CASIADataset(data[key], root=data.get("root", "."),
                            image_size=tuple(data["image_size"]))
                for key in ("train_manifest", "val_manifest")]
    if min(data["image_size"]) < 16:
        raise ValueError("U-Net requires image dimensions of at least 16")
    validate_split_isolation(*datasets)
    fingerprints = [_fingerprint(dataset) for dataset in datasets]
    loaders = [DataLoader(dataset, batch_size=training["batch_size"], shuffle=(i == 0),
                          num_workers=training.get("num_workers", 0),
                          generator=generator if i == 0 else None,
                          pin_memory=device.type == "cuda")
               for i, dataset in enumerate(datasets)]
    model = UNet(in_channels=3, out_channels=1,
                 base_channels=model_config.get("base_channels", 16)).to(device)
    loss_config = training["loss"]
    criterion = CombinedBCEDiceLoss(loss_config.get("bce_weight", 1.0),
                                    loss_config.get("dice_weight", 1.0),
                                    loss_config.get("smooth", 1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    checkpoint_dir = Path(config["paths"]["checkpoints_dir"])
    history_path = Path(config["paths"]["outputs_dir"]) / "history.csv"
    last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    history, start_epoch, best_dice, best_epoch, best_state = [], 1, -1.0, 0, None
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("kind") != "training_resume_v1":
            raise ValueError("Resume requires last.pt (full training state), not best.pt")
        if _resume_settings(saved["config"]) != _resume_settings(config):
            raise ValueError("Resume configuration mismatch (only epochs/device/output paths may change)")
        if saved["dataset_fingerprints"] != fingerprints:
            raise ValueError("Dataset changed since the checkpoint")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        stopper = EarlyStopping(**saved["early_stopping"])
        history, start_epoch = saved["history"], saved["epoch"] + 1
        best_dice, best_epoch, best_state = saved["best_dice"], saved["best_epoch"], saved["best_model_state"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        generator.set_state(saved["loader_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            if len(saved["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("CUDA device count changed; resume on CPU or the original GPU setup")
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        # Only append in the original location if its checkpoint agrees with the
        # requested resume point. Otherwise use new output directories.
        if last_path.exists():
            current = torch.load(last_path, map_location="cpu", weights_only=True)
            if (current.get("history") != history or
                    current.get("dataset_fingerprints") != fingerprints or
                    _resume_settings(current["config"]) != _resume_settings(config)):
                raise ValueError("Output directory contains a different run; choose new output paths")
        elif best_path.exists() or history_path.exists():
            raise FileExistsError("Resume destination contains artifacts without last.pt; choose new output paths")
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as stream:
                existing_history = list(csv.DictReader(stream))
            expected_history = [{k: str(v) for k, v in row.items()} for row in history]
            # A stale prefix is recoverable if a crash happened after last.pt was saved.
            if existing_history != expected_history[:len(existing_history)]:
                raise ValueError("History CSV belongs to a different run or resume point")
        if training["epochs"] < saved["epoch"]:
            raise ValueError("Total epochs cannot be less than the resumed epoch")
    elif any(path.exists() for path in (last_path, best_path, history_path)):
        raise FileExistsError("Run artifacts already exist; use --resume or new output directories")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def save_best():
        _atomic_save({"kind": "best_weights_v1", "model_state": best_state, "config": config,
                      "epoch": best_epoch, "val_dice": best_dice}, best_path)

    if history:
        if not last_path.exists():
            _atomic_save(saved, last_path)
        _write_history(history, history_path)
        save_best()
    print(f"Device: {device}; train samples: {len(datasets[0])}; validation samples: {len(datasets[1])}", flush=True)
    if stopper.bad_epochs >= stopper.patience:
        print("Checkpoint already reached early stopping; no further epochs run.", flush=True)
        return history
    for epoch in range(start_epoch, training["epochs"] + 1):
        train_result = train_one_epoch(model, loaders[0], optimizer, criterion, device, threshold)
        val_result = validate_one_epoch(model, loaders[1], criterion, device, threshold)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_result.items()},
               **{f"val_{k}": v for k, v in val_result.items()}}
        history.append(row)
        improved = val_result["dice"] > best_dice
        if improved:
            best_dice, best_epoch = val_result["dice"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        stopped = stopper.update(val_result[monitor])
        payload = {
            "kind": "training_resume_v1", "epoch": epoch, "config": config,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "best_model_state": best_state, "best_dice": best_dice, "best_epoch": best_epoch,
            "early_stopping": asdict(stopper), "history": history,
            "dataset_fingerprints": fingerprints, "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(), "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        # last.pt is authoritative; CSV and best.pt can be rebuilt on resume.
        _atomic_save(payload, last_path)
        save_best()
        _write_history(history, history_path)
        print(json.dumps(row), flush=True)
        if stopped:
            print(f"Early stopping after epoch {epoch} (validation {monitor}).", flush=True)
            break
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", type=Path)
    parser.add_argument("--resume", type=Path, help="Path to last.pt; epochs is the total epoch target")
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return run_training(config, args.resume)


if __name__ == "__main__":
    main()
