# Project overview: what this does and achieves

## What it does

1. **Downloads** the [Fruit Freshness Dataset v1](https://www.kaggle.com/datasets/user2036/fruit-freshness-dataset-v1) (Apple, Banana, Orange – fresh and rotten) via `kagglehub`.
2. **Trains** a model on the `train` folder: **80%** for training, **20%** for validation (random split, fixed seed). Default architecture is **ResNet-18 with ImageNet weights** (`python baseline.py`); use `--backbone small_cnn` for the original scratch CNN.
3. **Data:** Training uses **augmentation** (random crop, flip, rotation, color jitter). Validation and test use a fixed resize + ImageNet normalization (no augmentation).
4. **Training loop:** **30 epochs** with **Adam** and a **cosine learning-rate schedule**. ResNet uses **two learning rates** (lower on the trunk, higher on the final layer). The **best** weights by validation accuracy are saved to `best_model.pt` (including `arch`: `resnet18` or `small_cnn`) and reloaded for final metrics.
5. **Evaluates** and reports (on validation, and on the official **test** split when `dataset_root/test/` exists):
   - Accuracy  
   - Per-class precision, recall, F1 (and macro averages)  
   - Confusion matrix  
   - Multi-class ROC-AUC (one-vs-rest macro)

So: **download → train → save best checkpoint → print standard classification metrics** on val and test when available.

---

## What it achieves

- **Typical performance:** On the Kaggle layout with `train` + `test`, the scratch small CNN often reaches **~93% accuracy** and **macro AUC ~0.99+**. **ResNet-18 pretrained** is usually **similar or better**; exact numbers vary by backbone, run, and hardware.
- **Reproducible setup:** Dependencies and env are managed with **uv** (`pyproject.toml` + `uv.lock` + `.venv`), so the same versions and environment can be recreated.
- **Clarity on metrics:** The script and `METRICS_WALKTHROUGH.md` explain what each metric means (accuracy, precision, recall, F1, confusion matrix, ROC, AUC).

---

## What it does *not* do (yet)

- Does **not** use auxiliary losses (e.g. separate “fruit type” vs “freshness” heads) or custom penalties for fresh-vs-rotten confusions.

---

**In one line:** It downloads the fruit freshness data, trains an augmented classifier (default: **pretrained ResNet-18**) with a cosine LR schedule, saves the best checkpoint, and reports standard classification metrics on validation and (when present) the official test split.
