# Deep Learning-Based Image Forgery Detection and Localization

A pre-thesis research project using PyTorch, U-Net and CASIA v2 to predict where an image was manipulated.

**Status: the RGB training and validation pipeline is implemented**, including U-Net, validated datasets, BCE + Dice loss, pixel metrics, early stopping and resumable checkpoints. Standalone audit/evaluate/predict commands remain scaffolds. No real CASIA training or research results are included; synthetic tests only verify software behavior.

## Research objective

Build and evaluate an RGB-only segmentation baseline before investigating ELA, noise residuals, frequency-domain inputs or Transformer segmentation models. Establish trustworthy image-mask pairs and leakage-aware splits before comparing methods.

## Classification versus localization

Classification assigns an image-level label such as authentic or tampered. Localization predicts a label for each pixel, producing a manipulation mask. Real/Fake labels alone cannot supervise localization: tampered images require verified pixel-level ground truth.

## Architecture

The planned flow is:

```text
Configured image directories + verified masks
    -> dataset audit -> grouped train/validation/test manifests
    -> paired RGB/mask preprocessing -> PyTorch Dataset/DataLoader
    -> U-Net -> single-channel logits
    -> BCEWithLogitsLoss + Dice loss during training
    -> sigmoid probabilities -> thresholded masks during evaluation
```

Images have shape `(3, 256, 256)` and masks `(1, 256, 256)`. Batches add a leading batch dimension. Masks use 0 for authentic pixels and 1 for manipulated pixels. U-Net emits raw logits; sigmoid converts them to probabilities for Dice and inference.

## Folders and files

| Path | Purpose |
| --- | --- |
| `configs/baseline.yaml` | Editable dataset paths and baseline settings |
| `data/raw/` | Locally supplied authentic and tampered images |
| `data/masks/` | Verified tampered-image ground-truth masks |
| `data/processed/` | Future derived preprocessing artifacts |
| `data/splits/` | Future audited train/validation/test manifests |
| `notebooks/` | Exploratory analysis and Colab notebooks |
| `src/forgery_localization/datasets/` | Validated CSV image-mask loader and source-group split helpers |
| `src/forgery_localization/models/` | Implemented lightweight U-Net; space for later models |
| `src/forgery_localization/training/` | BCE/Dice losses, epoch loops and early stopping |
| `src/forgery_localization/evaluation/` | Pixel precision, recall, F1, Dice and IoU |
| `src/forgery_localization/utils/` | Dataset visualization and future reproducibility helpers |
| `scripts/` | Working train command; audit, evaluate and predict remain scaffolds |
| `tests/` | Synthetic dataset, visualization, import and U-Net tests; no CASIA data or GPU needed |
| `checkpoints/` | Best and resumable model checkpoints; ignored by Git |
| `outputs/` | Training history, synthetic checks and figures; ignored by Git |
| `reports/` | Future generated audit/evaluation reports; ignored by Git |
| `pyproject.toml` | Canonical package metadata and dependencies |
| `requirements.txt` | Editable install referencing the canonical dependencies |
| `.python-version` | Python 3.11 selection hint for environment managers |

## Environment setup and dependency installation

Use Python **3.11**. The version file does not install Python. Run commands from the repository root.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

Alternatively, `python -m pip install -e ".[dev]"` installs the same development environment. `python -m pip install -e .` omits pytest. The `src/` layout relies on an installed package for future scripts; use absolute imports such as `from forgery_localization.models import unet`. Pytest also has a local `src` path configured for scaffold smoke tests.

PyTorch, NumPy, Pillow, headless OpenCV, Matplotlib, scikit-learn and PyYAML are runtime dependencies. Headless OpenCV provides `cv2` without desktop GUI requirements, making it suitable for notebooks and servers. Dependency ranges are compatibility constraints, not a reproducible lockfile; record resolved versions with future experiments.

For a local GPU, install a PyTorch build matching your GPU driver and CUDA environment, then install this package. `device: auto` selects CUDA if available and CPU otherwise; explicit `cpu` and `cuda` settings are supported.

For Google Colab, place or clone this repository into the runtime, check `!python --version`, change into the repository root using `%cd`, and run `%pip install -e ".[dev]"`. This scaffold targets Python 3.11; if the runtime uses another version, select/provision a compatible runtime before installing. Mount Google Drive only if needed and configure absolute dataset/output paths there. Select a GPU runtime for later training; smoke tests do not require one. No code depends on a Windows-specific path.

## Dataset requirements

Download or obtain CASIA v2 separately under its applicable terms. The project assumes nothing is already downloaded. Set `authentic_dir`, `tampered_dir`, `masks_dir`, `processed_dir` and the three manifest paths in the YAML. Relative paths are intended to resolve from the repository root; absolute paths can point to external storage.

- Authentic images will use all-zero masks.
- Each tampered image must have a verified corresponding ground-truth manipulation mask. Image-level labels are insufficient.
- Do not invent masks or silently assign zero masks to tampered images with missing annotations. Exclude unresolved pairs from supervised localization and record the reason during auditing.
- Record mask provenance and check alignment, dimensions and binary encoding. Resize masks using nearest-neighbor interpolation and apply spatial augmentation jointly to the image and mask.

### Prepare an explicit manifest

Create a UTF-8 CSV with these required columns (paths below are illustrative and must refer to your own files):

```csv
image_path,mask_path,label,source_group,mask_verified,mask_source
data/raw/Au/original.png,,Au,source_001,,
data/raw/Tp/edited.png,data/masks/edited_mask.png,Tp,source_001,true,reviewed_annotation_release_and_record
```

`Au`/`authentic` and `Tp`/`tampered` are accepted, case-insensitively. Directory names and mask filenames are not inferred. Set `root` to the directory against which relative manifest and file paths should resolve; absolute paths are also accepted. The example paths are not a claim about the naming or organization of any CASIA distribution.

For every tampered row, supply a verified mask, `mask_verified=true`, and a nonempty `mask_source` identifying the annotation release or review record. This field records your attestation; the loader cannot independently verify provenance or whether a same-size mask describes the correct manipulated region. Authentic rows must leave `mask_path` empty and receive zeros.

Validation occurs at construction, decoding every listed image and mask. The loader rejects unreadable/missing files, duplicate image paths, reused mask paths, unknown labels, missing groups, mismatched original dimensions, nonbinary or colored/alpha masks, and tampered masks with no foreground. Masks may use grayscale 0/1 or 0/255 (including bilevel and grayscale RGB/palette encodings); 1/255 means manipulated. Other conventions must be reviewed and converted explicitly before loading. Non-default EXIF orientation is rejected so images cannot rotate independently from their annotations.

RGB images are converted to three channels, resized with bilinear interpolation, and scaled to float32 [0,1]. Masks use nearest-neighbor interpolation and return float32 0/1. If resizing removes all annotated foreground, access fails explicitly rather than returning an authentic-looking target. Files must remain unchanged after the initial audit. No augmentation is applied.

### Load samples and inspect them

After installing the package, with a manifest you have prepared:

```python
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from forgery_localization.datasets.casia_dataset import CASIADataset, split_dataset
from forgery_localization.utils.visualization import visualize_sample

root = Path.cwd()  # Repository root, or another explicit data root.
dataset = CASIADataset("data/splits/all.csv", root=root, image_size=(256, 256))
image, mask = dataset[0]
print(image.shape, mask.shape)  # (3,256,256), (1,256,256); both float32
figure = visualize_sample(image, mask)
plt.show()
plt.close(figure)

train_set, val_set, test_set = split_dataset(dataset, ratios=(0.7, 0.15, 0.15), seed=42)
loader = DataLoader(train_set, batch_size=8, shuffle=False, num_workers=0)
images, masks = next(iter(loader))
```

The split is reproducible for an unchanged manifest and seed. The trainer separately seeds shuffling and stores its RNG state in resumable checkpoints; augmentation is not implemented. At least three source groups are required. Ratios approximate **group counts**, not image counts, and each split has at least one group. Class stratification is not implemented: inspect sample counts, class balance and manipulation types before accepting a split. Preserve the manifest and split indices for experiment reproducibility.

Assign one `source_group` to the entire connected set of related originals, donor images, edits and known near-duplicates. Exact decoded RGB duplicates assigned conflicting labels or groups are rejected within a manifest. The loader cannot discover all donor relationships, recompressed copies or crops; source-aware manual auditing is still necessary. A random filename split is insufficient.

If you already have separate manifests, check them together before use:

```python
from forgery_localization.datasets.casia_dataset import CASIADataset, validate_split_isolation

train = CASIADataset("data/splits/train.csv", root=".")
val = CASIADataset("data/splits/val.csv", root=".")
test = CASIADataset("data/splits/test.csv", root=".")
validate_split_isolation(train, val, test)
```

This check rejects overlapping source groups, resolved image/mask paths and identical decoded image content across splits. It cannot establish absence of semantic leakage from unknown sources. Perform this check whenever loading separately curated splits; checking one dataset alone cannot detect overlap with another.

YAML integration is explicit, with no directory scanning or automatic pairing:

```python
from pathlib import Path
import yaml
from forgery_localization.datasets.casia_dataset import CASIADataset

config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
settings = config["dataset"]
train = CASIADataset(settings["train_manifest"], root=settings["root"],
                     image_size=tuple(settings["image_size"]))
```

The `authentic_dir`, `tampered_dir` and `masks_dir` YAML entries describe where you keep source files while curating the manifest; the manifest's explicit paths determine actual pairs. Validation and verified-mask requirements cannot be disabled through YAML. No real CASIA files have been tested here: annotation availability, mask polarity/alignment, release-specific file formats and source relationships remain unverified. Synthetic tests verify code behavior, not dataset quality.

Dataset directories, manifests and generated artifacts are ignored by Git; `.gitkeep` files preserve the folders. Keep any future shareable experiment documentation outside ignored artifact directories or deliberately revise the policy.

## Planned workflow and implementation checklist

1. [x] Implement U-Net with configurable channels and shape/gradient tests.
2. [ ] Audit CASIA v2 and verify tampered image-mask correspondences and provenance.
3. [ ] Curate real source groups/manifests and audit splits (seeded group split helpers implemented).
4. [x] Implement Dataset, DataLoader compatibility, paired resizing and zero authentic masks. Augmentation remains future work.
5. [x] Implement BCEWithLogitsLoss plus Dice loss, including empty-mask behavior.
6. [x] Implement seeding, auto device selection, AdamW training and validation.
7. [x] Implement and synthetically test best/last checkpoint saving and resuming. Real CASIA training remains pending.
8. [x] Implement pixel metrics. Dataset visualization is available; prediction CLI and final evaluation reports remain pending.
9. [ ] Run the RGB baseline and evaluate the held-out test set once decisions are fixed.
10. [ ] Record configurations, dependency versions, limitations and genuine results before extending the baseline.

Initial settings: 256 x 256 RGB inputs, 1 output channel, batch size 8, 30 epochs, learning rate 0.0001 and seed 42. The train command loads these settings from YAML and passes them to the existing model and loader.

## Using the U-Net

After installing the package in the environment above, use:

```python
import torch
from forgery_localization.models.unet import UNet

# Defaults: 3 RGB channels, 1 output channel, base width 16.
model = UNet()
model.eval()
images = torch.randn(2, 3, 256, 256)  # Synthetic inputs for a shape check.
with torch.no_grad():
    logits = model(images)
    probabilities = torch.sigmoid(logits)
    masks = probabilities >= 0.5
print(logits.shape)  # torch.Size([2, 1, 256, 256])
```

These randomly initialized outputs are not meaningful forgery predictions. Supply properly prepared RGB tensors and train on verified masks in a future step. To use a GPU, move both model and input tensors to the same CUDA device; the model itself is device-independent.

The encoder uses four double-convolution blocks with widths 16, 32, 64 and 128. Max pooling reduces spatial size from 256 to 16 across four levels, and the bottleneck has 256 channels. Each decoder block uses bilinear upsampling, a channel projection, an encoder skip connection and two convolutions. A final 1 x 1 convolution produces one logit per pixel. Padded convolutions preserve spatial dimensions; explicit resizing to skip dimensions also supports odd sizes. Height and width must each be at least 16.

`UNet(in_channels=3, out_channels=1, base_channels=16)` exposes the channel settings. The small default width and parameter-free bilinear resizing keep the baseline lightweight. There is no normalization layer, so batch size 1 also works at the minimum spatial size.

A synthetic backward-pass check (no dataset and no optimizer update):

```python
model.train()
model.zero_grad(set_to_none=True)
images = torch.randn(1, 3, 256, 256)
targets = torch.randint(0, 2, (1, 1, 256, 256)).float()
logits = model(images)
loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
loss.backward()
assert model.encoder1.layers[0].weight.grad is not None
```

Pass raw logits directly to `BCEWithLogitsLoss`; it includes sigmoid internally. The combined objective is available as `CombinedBCEDiceLoss` from `forgery_localization.training.losses`. The masks above are synthetic unit-check inputs, not annotations for real tampered images.

## Training and validation

First install the project using the Python 3.11 environment instructions above. Prepare verified, source-disjoint `train.csv` and `val.csv` manifests. Set `dataset.root`, `train_manifest`, `val_manifest` and `image_size` in `configs/baseline.yaml`. Relative paths resolve from the working directory (run from the repository root), not from the YAML file's location. Train and validation use identical RGB normalization, bilinear image resizing and nearest-neighbor mask resizing; only train batches are shuffled.

```bash
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/baseline.yaml --resume checkpoints/last.pt
```

In PowerShell, use `.\.venv\Scripts\python.exe` in place of `python` if the environment is not activated. Defaults are RGB U-Net with base width 16, AdamW, learning rate 1e-4, weight decay 0.01, batch size 8, 30 epochs, seed 42 and threshold 0.5. Edit a copy of the YAML for a separate experiment and give it separate checkpoint/output directories. `epochs` is the **total** target epoch count, not additional epochs on resume.

The command checks train/validation isolation before optimizing. It never reads or changes `test_manifest`; audit the frozen test split's source relationships separately during dataset preparation, and reserve test metrics for the final fixed model. No dataset or annotation downloads occur.

### Losses and metric conventions

The default objective is `1.0 * BCEWithLogitsLoss + 1.0 * SoftDiceLoss`. BCE uses raw logits and its numerically stable internal sigmoid. Soft Dice applies sigmoid and averages per-image Dice losses, with `smooth: 1.0` in numerator and denominator. An empty target and zero foreground probability have Dice loss zero; false foreground is penalized. BCE provides useful gradients for authentic images even when soft Dice saturates. Loss inputs must be finite, shape-matched and binary-targeted.

Epoch loss is weighted by the number of images in each batch. Pixel precision, recall, F1, Dice and IoU use thresholded sigmoid probabilities and aggregate TP/FP/FN across the whole epoch (**micro** metrics); F1 and Dice are identical under this definition. When both ground truth and predictions are entirely empty, every metric is 1. Otherwise undefined precision/recall is 0. Empty authentic images therefore do not inflate micro Dice through per-image averaging. Train metrics describe predictions collected as weights change; validation metrics use fixed weights in eval mode with no gradients. Class-specific/subset reports remain future work.

### Checkpoints, history and early stopping

| Default file | Contents |
| --- | --- |
| `checkpoints/best.pt` | Best validation-Dice model weights, selected epoch, Dice and configuration; inference only |
| `checkpoints/last.pt` | Latest model/optimizer state, best weights, epoch, history, early-stopping state, dataset fingerprints and RNG states; use for resume |
| `checkpoints/config.yaml` | Configuration used for the run |
| `outputs/history.csv` | Epoch, train/validation loss, precision, recall, F1, Dice and IoU |

Each epoch prints the same loss/metric values to the console. Checkpoint/history writes use temporary files followed by replacement. The full `last.pt` is authoritative if an interrupted write leaves the CSV or best checkpoint behind; resume reconstructs them. Fresh runs reject existing artifacts rather than silently replacing results.

`training.early_stopping.monitor` accepts `dice` (maximize) or `loss` (minimize). `patience` counts consecutive non-improving epochs and `min_delta` defines a meaningful improvement. Defaults are Dice, patience 10 and delta 0. `best.pt` always uses the highest validation Dice, independently of the early-stop monitor; ties retain the earlier epoch. Resuming a run that already reached early stopping does not restart optimization.

Resume allows changing total epochs, device and output paths. Other configuration changes or changed training/validation content are rejected to prevent mixing experiments. Epoch-boundary resume restores optimizer and Python/PyTorch/shuffle state. CPU resume equivalence is tested; identical results across GPU hardware, PyTorch versions or CPU/CUDA transitions are not promised. Only load checkpoints you trust. To load best weights:

```python
import torch
from forgery_localization.models.unet import UNet

saved = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=True)
settings = saved["config"]["model"]
model = UNet(in_channels=settings["in_channels"], out_channels=settings["out_channels"],
             base_channels=settings.get("base_channels", 16))
model.load_state_dict(saved["model_state"])
model.eval()
```

### Smoke tests and troubleshooting

```bash
python -m pytest tests/test_training.py -q
python -m pytest -q
```

A local full-size smoke run also completed two epochs (six optimizer steps) using 256 x 256 inputs and the default base width 16, resumed between epochs, and reloaded the best checkpoint. Its ignored artifacts are in `outputs/synthetic_training_demo/` (`verification.json`, `run.log`, `outputs/history.csv`, and `checkpoints/{best,last}.pt`). This run used Python 3.14.4 and PyTorch 2.15.0+cpu; Python 3.11 and CUDA were unavailable and remain unverified. The project continues to target Python 3.11.

Training tests create clearly labeled synthetic PNG images/masks in temporary directories, train for multiple batches, reload checkpoints, test resume against uninterrupted training, and run the CLI. They use smaller inputs/base width for speed while preserving the U-Net architecture. Existing model tests cover 256 x 256 inputs. Synthetic scores are software diagnostics, not research results.

- `No module named forgery_localization`: install with `python -m pip install -e ".[dev]"` in the same Python 3.11 environment used to run the command.
- Missing manifest/mask or invalid pairing: correct the CSV paths and audited annotations; no tampered-mask fallback exists.
- Cross-split overlap: repair source groups and split membership before training. Do not disable the check.
- CUDA unavailable: use `device: cpu`, or install a compatible CUDA-enabled PyTorch environment. CPU is slower but supported.
- Out-of-memory: lower `training.batch_size`; changing training settings requires a new run/output directory.
- DataLoader errors on Windows/notebooks: keep `num_workers: 0` initially and launch training through the guarded script entrypoint.
- Nonfinite loss/gradients: inspect input values and configuration. The trainer aborts before applying a nonfinite gradient update.
- Existing outputs or resume mismatch: resume the matching `last.pt`, or use new output directories; `best.pt` is not a full resume checkpoint.
- Foreground disappears on resizing: review tiny manipulated regions and image-size choice; do not silently convert these samples into authentic examples.

Before real CASIA training, obtain permitted images and verified masks, inspect polarity and alignment, establish source/donor groups, freeze audited splits, review class balance and sample counts, configure storage/device capacity, and run a small real-data sanity check. No actual CASIA files, mask release conventions or source relationships have been verified in this workspace. A final standalone test-evaluation/prediction command remains to be implemented before reporting thesis results.

## Verification

```bash
python -m compileall -q src scripts tests
python -m pytest -q
python -c "from pathlib import Path; import yaml; c = yaml.safe_load(Path('configs/baseline.yaml').read_text()); assert c['model']['in_channels'] == 3; print('YAML OK')"
git ls-files
```

Import tests exercise every module without accessing data. Synthetic dataset tests check decoding, interpolation, tensor shapes, invalid pairs, group isolation, duplicate detection and visualization. Model tests check batch sizes 1 and 2 at 256 x 256, finite outputs, raw logits, gradients through every parameter and the input, configurable channels, odd/minimum sizes and invalid inputs. These synthetic checks do not measure localization accuracy. Training tests also check loss stability, metric edge cases, parameter updates, validation isolation, early stopping, CLI execution and checkpoint resume equivalence. No real CASIA accuracy is inferred from these checks.
