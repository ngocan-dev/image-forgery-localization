"""Train one (architecture, input mode, seed) run with the controlled_multiseed_v1 protocol.

Reproduces the Colab protocol used for the U-Net-ResNet34 RGB/Noise/DCT runs
so that new architectures (SegFormer) are compared under identical settings:
batch 8, 10 epochs, AdamW lr 1e-4 wd 1e-4, weighted BCE (pos_weight 3) +
batch Dice in float32, fp16 autocast, deterministic paired flips
(sha256(seed, epoch, index)), checkpoint selection by validation micro
Dice@0.5 with lower validation loss as tie-break. The test set is never read.

    python scripts/train_multiinput.py --architecture segformer_mit_b2 --mode rgb --seed 7 \
        --train-manifest train.csv --val-manifest val.csv --root / \
        --output-root /content/drive/MyDrive/ImageForgeryResearch/checkpoints/topic2_v1

Output: ``<output-root>/<architecture>_<mode>_seed<seed>/`` with config.json,
last.pt (resumable), best_dice.pt, history.csv, validation_thresholds.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from forgery_localization.datasets.casia_dataset import CASIADataset, validate_split_isolation
from forgery_localization.features import MODES, FeatureBuilder, input_channels
from forgery_localization.models.factory import ARCHITECTURES, build_model

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]


def derived_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


class MultiInputDataset(Dataset):
    """Paired deterministic flips, then RGB normalization + optional extra features."""

    def __init__(self, base: CASIADataset, mode: str, seed: int, augment: bool) -> None:
        self.base, self.seed, self.augment, self.epoch = base, seed, augment, 0
        self.features = FeatureBuilder(mode)

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index):
        image, mask = self.base[index]
        if self.augment:
            generator = torch.Generator().manual_seed(derived_seed("augmentation", self.seed, self.epoch, index))
            horizontal, vertical = (torch.rand(2, generator=generator) < 0.5).tolist()
            if horizontal:
                image, mask = image.flip(-1), mask.flip(-1)
            if vertical:
                image, mask = image.flip(-2), mask.flip(-2)
        return self.features(image), mask.float().contiguous()


def loss_fn(logits, masks, pos_weight: float):
    logits, masks = logits.float(), masks.float()
    bce = F.binary_cross_entropy_with_logits(logits, masks, pos_weight=logits.new_tensor([pos_weight]))
    probabilities = logits.sigmoid()
    intersection = (probabilities * masks).sum()
    dice_loss = 1 - (2 * intersection + 1) / (probabilities.sum() + masks.sum() + 1)
    return bce + dice_loss


def run_epoch(model, loader, records, device, pos_weight, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    thresholds = [0.5] if training else THRESHOLDS
    stats = {t: dict(tp=0, fp=0, fn=0, dice_sum=0.0, n_tampered=0, auth_fp=0, auth_pixels=0)
             for t in thresholds}
    total_loss, count = 0.0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, masks, pos_weight)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Loss NaN/Inf; epoch not saved.")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            with torch.no_grad():
                probabilities = logits.float().sigmoid()
                target = masks >= 0.5
                tampered = target.flatten(1).any(1)
                if not training:
                    expected = torch.tensor([r["label"] == "tampered" for r in records[count:count + len(images)]],
                                            device=device)
                    assert torch.equal(tampered, expected), "Validation label/mask mismatch"
                for threshold, s in stats.items():
                    predicted = probabilities >= threshold
                    tp = (predicted & target).sum((1, 2, 3))
                    fp = (predicted & ~target).sum((1, 2, 3))
                    fn = (~predicted & target).sum((1, 2, 3))
                    s["tp"] += tp.sum().item()
                    s["fp"] += fp.sum().item()
                    s["fn"] += fn.sum().item()
                    if not training:
                        dice = 2 * tp.float() / (2 * tp + fp + fn).clamp_min(1)
                        s["dice_sum"] += dice[tampered].sum().item()
                        s["n_tampered"] += tampered.sum().item()
                        s["auth_fp"] += predicted[~tampered].sum().item()
                        s["auth_pixels"] += predicted[~tampered].numel()
            total_loss += loss.item() * len(images)
            count += len(images)
    rows = []
    for threshold, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        row = {"threshold": threshold, "micro_dice": 2 * tp / max(2 * tp + fp + fn, 1),
               "micro_iou": tp / max(tp + fp + fn, 1), "pixel_precision": tp / max(tp + fp, 1),
               "pixel_recall": tp / max(tp + fn, 1)}
        if not training:
            row["mean_tampered_dice"] = s["dice_sum"] / max(s["n_tampered"], 1)
            row["authentic_fp_pixel_pct"] = 100 * s["auth_fp"] / max(s["auth_pixels"], 1)
        rows.append(row)
    return total_loss / count, rows


def _save(payload, path: Path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _csv(rows, path: Path):
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def signature(dataset: CASIADataset) -> str:
    text = json.dumps([[r["image_hash"], r["label"], r["source_group"]] for r in dataset.records])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required (the protocol uses fp16 autocast).")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_base = CASIADataset(args.train_manifest, root=args.root, image_size=(256, 256))
    val_base = CASIADataset(args.val_manifest, root=args.root, image_size=(256, 256))
    validate_split_isolation(train_base, val_base)

    import segmentation_models_pytorch as smp
    config = {
        "protocol": "controlled_multiseed_v1",
        "architecture": args.architecture,
        "encoder": ARCHITECTURES[args.architecture][1],
        "encoder_weights": "imagenet",
        "mode": args.mode,
        "seed": args.seed,
        "input_channels": input_channels(args.mode),
        "image_size": [256, 256],
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "pos_weight": args.pos_weight,
        "loss": "weighted_bce_plus_batch_dice_float32",
        "augmentation": "paired_horizontal_vertical_flip_p0.5",
        "normalization": "rgb_imagenet",
        "first_conv": "rgb_pretrained" if args.mode == "rgb" else "rgb_pretrained_extra_zero",
        "thresholds": THRESHOLDS,
        "selection": "val_micro_dice_0.5_then_lower_val_loss",
        "train_signature": signature(train_base),
        "val_signature": signature(val_base),
        "torch_version": str(torch.__version__),
        "smp_version": str(smp.__version__),
    }
    directory = Path(args.output_root) / f"{args.architecture}_{args.mode}_seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    config_path, last_path = directory / "config.json", directory / "last.pt"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise SystemExit(f"config.json differs from this command: {directory}")
    else:
        if list(directory.glob("*.pt")):
            raise SystemExit("Checkpoints exist without config.json")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    saved = torch.load(last_path, map_location="cpu", weights_only=True) if last_path.exists() else None
    if saved is not None and saved["epoch"] >= args.epochs:
        print(f"SKIP {directory.name}: already complete")
        return
    model = build_model(args.architecture, args.mode, pretrained=saved is None, seed=args.seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    train_data = MultiInputDataset(train_base, args.mode, args.seed, augment=True)
    val_data = MultiInputDataset(val_base, args.mode, args.seed, augment=False)
    shuffle_generator = torch.Generator()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=shuffle_generator,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    state = dict(history=[], threshold_history=[], best_model=None, best_epoch=None, best_metrics=None,
                 best_dice=-1.0, best_loss_at_dice=float("inf"))
    start_epoch = 1
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        for key in state:
            state[key] = saved[key]
        start_epoch = saved["epoch"] + 1
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
        print(f"RESUME {directory.name} at epoch {start_epoch}")

    def capture(epoch):
        return {"config": config, "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), **state,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}

    for epoch in range(start_epoch, args.epochs + 1):
        train_data.set_epoch(epoch)
        shuffle_generator.manual_seed(derived_seed("shuffle", args.seed, epoch))
        train_loss, train_rows = run_epoch(model, train_loader, None, device, args.pos_weight, optimizer, scaler)
        val_loss, val_rows = run_epoch(model, val_loader, val_base.records, device, args.pos_weight)
        val05 = next(row for row in val_rows if row["threshold"] == 0.5)
        score = val05["micro_dice"]
        if score > state["best_dice"] or (score == state["best_dice"] and val_loss < state["best_loss_at_dice"]):
            state.update(best_dice=score, best_loss_at_dice=val_loss, best_epoch=epoch,
                         best_metrics={"val_loss": val_loss, **val05},
                         best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        state["history"].append({"epoch": epoch, "train_loss": train_loss,
                                 "train_online_dice_05": train_rows[0]["micro_dice"], "val_loss": val_loss,
                                 **{f"val_{k}": v for k, v in val05.items() if k != "threshold"}})
        state["threshold_history"].extend({"epoch": epoch, **row} for row in val_rows)
        _save(capture(epoch), last_path)
        _csv(state["history"], directory / "history.csv")
        _csv(state["threshold_history"], directory / "validation_thresholds.csv")
        _save({"config": config, "epoch": state["best_epoch"], "model": state["best_model"],
               "selection_metrics": state["best_metrics"]}, directory / "best_dice.pt")
        print(f"{directory.name} | Epoch {epoch:02d}/{args.epochs} | Train loss {train_loss:.4f} | "
              f"Val loss {val_loss:.4f} | Dice@0.5 {score:.4f} | "
              f"Mean tampered Dice {val05['mean_tampered_dice']:.4f} | "
              f"Authentic FP px {val05['authentic_fp_pixel_pct']:.2f}%", flush=True)
    print(f"DONE {directory.name}: best epoch {state['best_epoch']}, Dice {state['best_dice']:.6f}")


if __name__ == "__main__":
    main()
