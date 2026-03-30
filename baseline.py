"""
Fruit freshness baseline: download dataset, train a CNN (scratch or pretrained
ResNet-18), and compute standard classification metrics with explanations.
Run: python baseline.py
     python baseline.py --backbone small_cnn
"""

import argparse
import os
import sys
from pathlib import Path
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# -----------------------------------------------------------------------------
# 1. DOWNLOAD DATASET & DISCOVER STRUCTURE
# -----------------------------------------------------------------------------

def get_data_path():
    print("Downloading dataset (may use cache)...")
    path = kagglehub.dataset_download("user2036/fruit-freshness-dataset-v1")
    path = Path(path)
    print(f"Dataset root: {path}")
    # Common layouts: path/train/fresh, path/train/rotten OR path/fresh, path/rotten
    for sub in path.iterdir():
        if sub.is_dir() and not sub.name.startswith("."):
            print(f"  {sub.name}/")
            for sub2 in list(sub.iterdir())[:5]:
                if sub2.is_dir():
                    print(f"    {sub2.name}/")
    return path


def _has_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for f in folder.iterdir():
        if f.suffix.lower() in exts:
            return True
    return False


def find_image_folders(root: Path):
    """Find folder that has class subfolders (each with images)."""
    root = Path(root)
    # Option A: root/train/fresh, root/train/rotten
    for split in ("train", "Train", "training"):
        train_dir = root / split
        if train_dir.is_dir():
            subdirs = [d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if subdirs and _has_images(subdirs[0]):
                return train_dir, None
    # Option B: root/fresh, root/rotten (or root/Apple_Fresh, etc.)
    subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    for d in subdirs:
        if _has_images(d):
            return root, None
    return None, "Could not find class subfolders with images"


# -----------------------------------------------------------------------------
# 2. DATA LOADERS
# -----------------------------------------------------------------------------

def get_train_transform(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),                          # ADD: handles upside-down
        transforms.RandomRotation(180),                           # CHANGE: was 15°, now full rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),          # INCREASE: more colour variety
        transforms.RandomGrayscale(p=0.05),                       # ADD: robustness
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size=224):
    """Deterministic transform for validation and test (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset so we can apply a different transform than the parent dataset."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


def build_dataloaders(data_dir, batch_size=32, image_size=224, val_ratio=0.2, seed=42, device=None):
    data_dir = Path(data_dir)
    full_ds = datasets.ImageFolder(str(data_dir), transform=None)
    n = len(full_ds)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_sub, val_sub = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_ds = _TransformSubset(train_sub, get_train_transform(image_size))
    val_ds = _TransformSubset(val_sub, get_val_transform(image_size))

    pin = device is not None and device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, full_ds.classes


def build_test_loader(test_dir, batch_size=32, image_size=224):
    """Build a DataLoader for the official test set. test_dir should be dataset_root / 'test'."""
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        return None, None
    test_ds = datasets.ImageFolder(str(test_dir), transform=get_val_transform(image_size))
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0), test_ds.classes


# -----------------------------------------------------------------------------
# 3. MODEL
# -----------------------------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_resnet18(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with ImageNet weights; final layer replaced for `num_classes`."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def build_model(backbone: str, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """`backbone`: 'resnet18' | 'small_cnn'. `pretrained` applies only to ResNet."""
    b = backbone.lower().strip()
    if b == "small_cnn":
        return SmallCNN(num_classes)
    if b == "resnet18":
        return build_resnet18(num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown backbone: {backbone!r}. Use 'resnet18' or 'small_cnn'.")


def build_optimizer(model: nn.Module, backbone: str):
    """Lower LR on pretrained ResNet trunk; higher on the classification head."""
    b = backbone.lower().strip()
    if b == "small_cnn":
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    if b == "resnet18":
        backbone_params = []
        head_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("fc."):
                head_params.append(p)
            else:
                backbone_params.append(p)
        return torch.optim.Adam(
            [
                {"params": backbone_params, "lr": 1e-4},
                {"params": head_params, "lr": 1e-3},
            ]
        )
    raise ValueError(f"Unknown backbone: {backbone!r}")


def load_model_from_checkpoint(
    path: str | Path,
    *,
    map_location=None,
) -> tuple[nn.Module, dict]:
    """
    Load `best_model.pt` and rebuild the correct architecture.
    Checkpoints from older runs without `arch` default to `small_cnn`.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)
    arch = ckpt.get("arch", "small_cnn")
    pretrained = False  # weights come from checkpoint
    model = build_model(arch, num_classes, pretrained=pretrained)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


# -----------------------------------------------------------------------------
# 4. TRAINING
# -----------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# -----------------------------------------------------------------------------
# 5. EVALUATION & METRICS (with explanations)
# -----------------------------------------------------------------------------

def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # for ROC-AUC
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    return y_true, y_pred, y_probs, class_names


def print_metrics(y_true, y_pred, y_probs, class_names):
    n_classes = len(class_names)

    # ----- (1) ACCURACY -----
    # What it means: Of all samples, what fraction did we get right?
    # Formula: correct_predictions / total_predictions
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 60)
    print("1. ACCURACY")
    print("=" * 60)
    print("Meaning: Of all images, the fraction we predicted correctly.")
    print("Formula: (number correct) / (total number of samples)")
    print(f"Accuracy = {acc:.4f}  ({acc*100:.2f}%)")
    print()

    # ----- (2) PRECISION, RECALL, F1 (per class) -----
    # Precision: Of all we predicted as class X, how many were really X? (avoids false alarms)
    # Recall:    Of all true X, how many did we find? (avoids missing real X)
    # F1:        Harmonic mean of precision and recall (single balance score per class)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), average=None, zero_division=0
    )
    print("2. PRECISION, RECALL, F1 (per class)")
    print("=" * 60)
    print("Precision: Of everything we predicted as this class, how many were actually this class?")
    print("           High precision = fewer false positives (we don't call rotten when it's fresh).")
    print("Recall:    Of all true samples of this class, how many did we correctly find?")
    print("           High recall = we don't miss many (we catch most rotten fruit).")
    print("F1:        Harmonic mean of precision and recall (balances both).")
    print()
    for i, name in enumerate(class_names):
        print(f"  {name}:")
        print(f"    Precision = {precision[i]:.4f}   Recall = {recall[i]:.4f}   F1 = {f1[i]:.4f}   (n = {support[i]})")
    # Macro average (treat each class equally)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), average="macro", zero_division=0
    )
    print(f"  Macro avg: Precision = {p_macro:.4f}   Recall = {r_macro:.4f}   F1 = {f_macro:.4f}")
    print()

    # ----- (3) CONFUSION MATRIX -----
    # Rows = true class, Cols = predicted class. Entry (i,j) = count of true i predicted as j.
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    print("3. CONFUSION MATRIX")
    print("=" * 60)
    print("Meaning: Rows = true label, Columns = predicted label.")
    print("Entry [i,j] = number of samples with true class i that we predicted as class j.")
    print("Diagonal = correct; off-diagonal = confusions (e.g. fresh predicted as rotten).")
    print()
    print("Predicted →")
    header = "True ↓  " + "  ".join(f"{c[:8]:>8}" for c in class_names)
    print(header)
    for i in range(n_classes):
        row = f"{class_names[i][:8]:>8}  " + "  ".join(f"{cm[i,j]:>8}" for j in range(n_classes))
        print(row)
    print()

    # ----- (4) ROC-AUC -----
    # Only valid for binary; for multi-class we use one-vs-rest or one-vs-one.
    if n_classes == 2:
        # Binary: use probability of positive class
        auc = roc_auc_score(y_true, y_probs[:, 1])
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        print("4. ROC-AUC (binary)")
        print("=" * 60)
        print("Meaning: AUC measures how well the model ranks positives above negatives.")
        print("We use the model's 'freshness score' (softmax prob for positive class).")
        print("AUC = 1.0: perfect ranking; 0.5: random; <0.5: worse than random.")
        print(f"AUC = {auc:.4f}")
        return {"fpr": fpr, "tpr": tpr, "auc": auc}
    else:
        # Multi-class: one-vs-rest AUC (macro)
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            print("4. ROC-AUC (multi-class, one-vs-rest macro)")
            print("=" * 60)
            print("Meaning: For each class we treat it as 'positive' vs rest; AUC per class then averaged.")
            print("Measures how well softmax probabilities separate each class from the others.")
            print(f"Macro AUC = {auc:.4f}")
        except Exception as e:
            print("4. ROC-AUC: skipped (multi-class issue)", e)
        return None


def plot_roc(fpr, tpr, auc, save_path="roc_curve.png"):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (freshness score)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"ROC curve saved to {save_path}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train fruit freshness classifier.")
    p.add_argument(
        "--backbone",
        choices=("resnet18", "small_cnn"),
        default="resnet18",
        help="Model: ResNet-18 + ImageNet (default) or small CNN from scratch.",
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Random ResNet-18 init instead of ImageNet weights (ignored for --backbone small_cnn).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.backbone == "resnet18":
        pre = "ImageNet pretrained" if not args.no_pretrained else "random init"
        print(f"Backbone: ResNet-18 ({pre})")
    else:
        print("Backbone: SmallCNN (from scratch)")

    data_path = get_data_path()
    data_dir, err = find_image_folders(data_path)
    if err:
        # Try using data_path directly as ImageFolder root
        data_dir = data_path
        if not any((data_path / d).is_dir() for d in os.listdir(data_path) if not d.startswith(".")):
            print(err)
            sys.exit(1)

    print(f"Using data directory: {data_dir}")
    train_loader, val_loader, class_names = build_dataloaders(data_dir, device=device)
    print(f"Classes: {class_names}")

    # Official test set (dataset_root/test)
    test_dir = data_dir.parent / "test"
    test_loader, test_class_names = build_test_loader(test_dir)
    if test_loader is not None:
        print(f"Test set: {test_dir} ({len(test_loader.dataset)} images)")
    else:
        print("No official test set found; reporting validation metrics only.")

    num_classes = len(class_names)
    pretrained = args.backbone == "resnet18" and not args.no_pretrained
    model = build_model(args.backbone, num_classes, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.backbone)

    num_epochs = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_acc = -1.0
    checkpoint_path = Path("best_model.pt")

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        # Validation accuracy for checkpointing
        y_true_v, y_pred_v, _, _ = evaluate(model, val_loader, device, class_names)
        val_acc = float(accuracy_score(y_true_v, y_pred_v))
        lr = optimizer.param_groups[0]["lr"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "arch": args.backbone,
            }, checkpoint_path)
            print(f"Epoch {epoch}/{num_epochs}  train_loss = {loss:.4f}  val_acc = {val_acc:.4f}  lr = {lr:.6f}  [saved best]")
        else:
            print(f"Epoch {epoch}/{num_epochs}  train_loss = {loss:.4f}  val_acc = {val_acc:.4f}  lr = {lr:.6f}")

    print(f"Best validation accuracy: {best_val_acc:.4f}  (checkpoint: {checkpoint_path})")

    # Load best checkpoint so reported metrics match the saved model
    ckpt = torch.load(checkpoint_path, weights_only=True)
    arch = ckpt.get("arch", "small_cnn")
    model = build_model(arch, num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best checkpoint (epoch {ckpt['epoch']}, arch={arch}) for evaluation.")

    # Validation set metrics (using best checkpoint)
    print("\n" + "=" * 60)
    print("VALIDATION SET (20% holdout from train) — best checkpoint")
    print("=" * 60)
    y_true, y_pred, y_probs, _ = evaluate(model, val_loader, device, class_names)
    roc_data = print_metrics(y_true, y_pred, y_probs, class_names)
    if roc_data:
        plot_roc(roc_data["fpr"], roc_data["tpr"], roc_data["auc"], save_path="roc_curve_val.png")

    # Official test set metrics (unbiased estimate, best checkpoint)
    if test_loader is not None:
        print("\n" + "=" * 60)
        print("TEST SET (official held-out split) — best checkpoint")
        print("=" * 60)
        y_true_t, y_pred_t, y_probs_t, _ = evaluate(model, test_loader, device, test_class_names)
        print_metrics(y_true_t, y_pred_t, y_probs_t, test_class_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
