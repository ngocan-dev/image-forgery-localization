"""Build and load segmentation models from segmentation_models_pytorch (smp).

Supported architectures
-----------------------
``unet_resnet34``     U-Net decoder + ImageNet ResNet34 encoder (CNN baseline,
                      identical to the Colab ``controlled_multiseed_v1`` runs).
``segformer_mit_b0``  SegFormer decoder + Mix Transformer encoder (Transformer
``segformer_mit_b2``  upgrade required by topic 2). b0 is light, b2 stronger.
``unet_mit_b2``       U-Net decoder + Mix Transformer encoder (hybrid option).

For 6-channel inputs the first convolution keeps its pretrained RGB weights
and the three extra channels start at zero ("rgb_pretrained_extra_zero"), so
at initialisation the model behaves exactly like its RGB counterpart.

With ``cls_head=True`` an image-level classification head (smp
``ClassificationHead``: global average pooling + dropout + linear on the
deepest encoder feature) is attached AFTER smp has initialised the model, so
the encoder/decoder start from exactly the same weights as the run without a
head (passing ``aux_params`` would draw the head first and shift the decoder
initialisation). The model then returns ``(mask_logits, image_logits)``; use
``split_outputs`` to handle both kinds of model uniformly.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from forgery_localization.features import input_channels

ARCHITECTURES = {
    "unet_resnet34": ("Unet", "resnet34"),
    "segformer_mit_b0": ("Segformer", "mit_b0"),
    "segformer_mit_b2": ("Segformer", "mit_b2"),
    "unet_mit_b2": ("Unet", "mit_b2"),
}


def _smp():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "segmentation-models-pytorch is required: pip install segmentation-models-pytorch"
        ) from error
    return smp


def _first_rgb_conv(module: nn.Module) -> tuple[nn.Module, str, nn.Conv2d]:
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d) and child.in_channels == 3:
            parent_name, _, attribute = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            return parent, attribute, child
    raise ValueError("No 3-channel input convolution found in encoder")


def expand_input_channels(model: nn.Module, channels: int) -> None:
    """Replace the encoder stem so it accepts ``channels`` inputs (extra = zero)."""
    if channels == 3:
        return
    parent, attribute, old = _first_rgb_conv(model.encoder)
    # Do not consume the CPU RNG: keeps model init identical to the RGB run.
    with torch.random.fork_rng(devices=[]):
        new = nn.Conv2d(
            channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, dilation=old.dilation, groups=old.groups,
            bias=old.bias is not None, padding_mode=old.padding_mode,
        )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(parent, attribute, new)
    if hasattr(model.encoder, "_in_channels"):
        model.encoder._in_channels = channels
    if hasattr(model.encoder, "_out_channels"):
        out = list(model.encoder._out_channels)
        out[0] = channels
        model.encoder._out_channels = tuple(out)


CLS_HEAD_PARAMS = {"classes": 1, "pooling": "avg", "dropout": 0.2}


def split_outputs(output) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(mask_logits (B,1,H,W), image_logits (B,1) or None)`` for any model built here."""
    if isinstance(output, (tuple, list)):
        return output[0], output[1]
    return output, None


def model_label(config: dict) -> str:
    """Architecture name used in reports; a training variant (e.g. v2) is appended."""
    architecture = config.get("architecture") or architecture_from_config(config)
    variant = config.get("variant")
    return f"{architecture}-{variant}" if variant else architecture


def build_model(architecture: str, mode: str, pretrained: bool = True,
                seed: int | None = None, cls_head: bool = False) -> nn.Module:
    """Return a model emitting (B, 1, H, W) logits (plus (B, 1) image logits with ``cls_head``)."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {sorted(ARCHITECTURES)}")
    smp = _smp()
    class_name, encoder = ARCHITECTURES[architecture]
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    model = getattr(smp, class_name)(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    expand_input_channels(model, input_channels(mode))
    if cls_head:
        from segmentation_models_pytorch.base import ClassificationHead
        from segmentation_models_pytorch.base.initialization import initialize_head
        with torch.random.fork_rng(devices=[]):   # leave the caller's RNG stream untouched
            head = ClassificationHead(in_channels=model.encoder.out_channels[-1], **CLS_HEAD_PARAMS)
            initialize_head(head)
        model.classification_head = head          # smp's forward then returns (masks, labels)
    return model


def architecture_from_config(config: dict) -> str:
    """Notebook configs only store ``encoder: resnet34``; repo configs store ``architecture``."""
    if "architecture" in config:
        return config["architecture"]
    if config.get("encoder") == "resnet34":
        return "unet_resnet34"
    raise ValueError("Checkpoint config does not identify the architecture")


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"
                    ) -> tuple[nn.Module, dict]:
    """Load a ``best_dice.pt`` written by the notebook or scripts/train_multiinput.py."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "config" not in payload or "model" not in payload:
        raise ValueError(f"{path} is not a best_dice.pt checkpoint with config and model")
    config = dict(payload["config"])
    config.setdefault("architecture", architecture_from_config(config))
    model = build_model(config["architecture"], config["mode"], pretrained=False,
                        cls_head=bool(config.get("cls_head", False)))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    config["checkpoint_epoch"] = payload.get("epoch")
    config["checkpoint_path"] = str(path)
    return model, config
