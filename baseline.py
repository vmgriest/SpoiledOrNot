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
from typing import Optional, Tuple, Union, Dict, List

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
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms


# -----------------------------------------------------------------------------
# 1. CLASS FILTERING (for dataset alignment)
# -----------------------------------------------------------------------------

def normalize_class_name(class_name: str) -> str:
    """Normalize class names to handle spelling variations."""
    # Common spelling corrections
    corrections = {
        'freshpatato': 'freshpotato',
        'freshtamto': 'freshtomato',
        'rottenpatato': 'rottenpotato',
        'rottentamto': 'rottentomato',
        'freshbittergroud': None,  # Remove these classes
        'freshcapsicum': None,
        'rottenbittergroud': None,
        'rottencapsicum': None,
    }
    return corrections.get(class_name, class_name)


class FilteredDataset(torch.utils.data.Dataset):
    """Dataset that filters and remaps classes."""
    
    def __init__(self, original_dataset, class_mapping, transform=None):
        """
        Args:
            original_dataset: The original ImageFolder dataset
            class_mapping: Dict mapping original label -> new label
            transform: Optional transform to apply
        """
        self.original_dataset = original_dataset
        self.class_mapping = class_mapping
        self.transform = transform
        
        # Create list of indices to keep (only those with mapping)
        self.indices = []
        for idx in range(len(original_dataset)):
            _, label = original_dataset[idx]
            if label in class_mapping:
                self.indices.append(idx)
        
        # Get the new class names in order
        inverse_mapping = {}
        for old, new in class_mapping.items():
            inverse_mapping[new] = old
        
        # Build class names list
        self.classes = []
        for new_idx in sorted(inverse_mapping.keys()):
            old_idx = inverse_mapping[new_idx]
            self.classes.append(original_dataset.classes[old_idx])
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, old_label = self.original_dataset[original_idx]
        new_label = self.class_mapping[old_label]
        
        if self.transform:
            img = self.transform(img)
            
        return img, new_label


# -----------------------------------------------------------------------------
# 2. DOWNLOAD DATASET & DISCOVER STRUCTURE
# -----------------------------------------------------------------------------

def get_data_path() -> Path:
    """Get the path to the dataset, downloading if necessary."""
    # First check if there is already a local data folder
    local_data = Path("./data")
    if local_data.exists():
        # Check if it has image subfolders
        subdirs = [d for d in local_data.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if subdirs and any(_has_images(d) for d in subdirs):
            print(f"Using local data folder: {local_data}")
            return local_data

    print("Downloading dataset (may use cache)...")

    # Try multiple datasets (in order of preference)
    datasets_to_try = [
        "swoyam2609/fresh-and-stale-classification",
        "user2036/fruit-freshness-dataset-v1",
    ]

    path: Optional[Path] = None
    last_error = None
    for dataset_id in datasets_to_try:
        try:
            print(f"  Trying dataset: {dataset_id}...")
            downloaded_path = kagglehub.dataset_download(dataset_id)
            path = Path(downloaded_path)
            print(f"Successfully downloaded: {dataset_id}")
            break
        except Exception as e:
            last_error = str(e)
            print(f"  Failed: {dataset_id}")
            continue

    if path is None:
        print("\n" + "=" * 60)
        print("Could not download dataset automatically.")
        print("=" * 60)
        print("\nTo use this project, you need to:")
        print("\nOption 1 - Manual Download (Recommended):")
        print("  1. Go to: https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification")
        print("  2. Click 'Download' button (top right)")
        print("  3. Extract the zip to: ./data/ folder in this project")
        print("  4. Run: python baseline.py")
        print("\nOption 2 - Set up Kaggle API:")
        print("  1. Get API token from kaggle.com/account")
        print("  2. Place kaggle.json in: C:\\Users\\<yourname>\\.kaggle\\")
        print("  3. Run: python baseline.py")
        print("=" * 60)
        print(f"\nError: {last_error[:100]}...")
        sys.exit(1)

    print(f"Dataset root: {path}")

    # Check if there is a 'dataset' subfolder (common in this dataset)
    dataset_subfolder = path / "dataset"
    if dataset_subfolder.exists() and dataset_subfolder.is_dir():
        print("Found 'dataset' subfolder, using that as root")
        path = dataset_subfolder

    # Common layouts: path/train/fresh, path/train/rotten OR path/fresh, path/rotten
    for sub in path.iterdir():
        if sub.is_dir() and not sub.name.startswith("."):
            print(f"  {sub.name}/")
            for sub2 in list(sub.iterdir())[:5]:
                if sub2.is_dir():
                    print(f"    {sub2.name}/")
    return path


def _has_images(folder: Path) -> bool:
    """Check if a folder contains any image files."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for f in folder.iterdir():
        if f.suffix.lower() in exts:
            return True
    return False


def find_image_folders(root: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Find folder that has class subfolders (each with images)."""
    root = Path(root)
    # Option A: root/train/fresh, root/train/rotten (check both lowercase and capitalized)
    for split in ("train", "Train", "training", "Train"):
        train_dir = root / split
        if train_dir.is_dir():
            subdirs = [d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if subdirs and _has_images(subdirs[0]):
                return train_dir, None

    # Also check for root/test or root/Test
    for split in ("test", "Test"):
        test_dir = root / split
        if test_dir.is_dir():
            subdirs = [d for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if subdirs and _has_images(subdirs[0]):
                return test_dir, None

    # Option B: root/fresh, root/rotten (or root/Apple_Fresh, etc.)
    subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    for d in subdirs:
        if _has_images(d):
            return root, None
    return None, "Could not find class subfolders with images"


# -----------------------------------------------------------------------------
# 2. DATA LOADERS
# -----------------------------------------------------------------------------

def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Get training data augmentation transforms optimized for spoilage detection."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.15),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.RandomPosterize(bits=4, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic transform for validation and test (minimal augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    image_size: int = 224,
    val_ratio: float = 0.2,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Build training and validation dataloaders with class alignment."""
    data_dir = Path(data_dir)
    
    # Load original dataset without transforms
    original_dataset = datasets.ImageFolder(str(data_dir), transform=None)
    
    # Create mapping from original labels to new labels
    class_mapping = {}
    new_idx = 0
    new_classes = []
    seen_classes = set()
    
    for old_idx, class_name in enumerate(original_dataset.classes):
        normalized = normalize_class_name(class_name)
        if normalized is not None:  # Keep only valid classes
            if normalized not in seen_classes:
                seen_classes.add(normalized)
                new_classes.append(normalized)
            class_mapping[old_idx] = new_classes.index(normalized)
    
    print(f"Original classes: {len(original_dataset.classes)}")
    print(f"Filtered classes: {len(new_classes)}")
    
    # Create filtered dataset
    filtered_dataset = FilteredDataset(original_dataset, class_mapping)
    
    # Split into train and validation
    n = len(filtered_dataset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    
    train_dataset, val_dataset = random_split(
        filtered_dataset, [n_train, n_val], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = get_train_transform(image_size)
    val_dataset.dataset.transform = get_val_transform(image_size)
    
    # Create data loaders
    pin = device is not None and device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, new_classes


def build_test_loader(
    test_dir: Union[str, Path],
    train_classes: List[str],
    batch_size: int = 32,
    image_size: int = 224
) -> Tuple[Optional[DataLoader], Optional[List[str]]]:
    """Build a DataLoader for the official test set with alignment to training classes."""
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        return None, None
    
    # Load test dataset
    original_test_dataset = datasets.ImageFolder(str(test_dir), transform=None)
    
    # Create mapping from test class names to training class indices
    train_name_to_idx = {name: idx for idx, name in enumerate(train_classes)}
    class_mapping = {}
    
    for old_idx, class_name in enumerate(original_test_dataset.classes):
        normalized = normalize_class_name(class_name)
        if normalized in train_name_to_idx:
            class_mapping[old_idx] = train_name_to_idx[normalized]
    
    if not class_mapping:
        print(f"Warning: No matching classes found between test and train")
        print(f"Test classes: {original_test_dataset.classes}")
        print(f"Train classes: {train_classes}")
        return None, None
    
    # Create filtered test dataset with transforms
    filtered_test_dataset = FilteredDataset(
        original_test_dataset, 
        class_mapping,
        transform=get_val_transform(image_size)
    )
    
    test_loader = DataLoader(
        filtered_test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return test_loader, train_classes


# -----------------------------------------------------------------------------
# 3. MODEL
# -----------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """Small CNN model for fruit freshness classification."""

    def __init__(self, num_classes: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Build a model with the specified backbone.

    Args:
        backbone: 'resnet18' or 'small_cnn'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (only for ResNet)

    Returns:
        The model instance
    """
    b = backbone.lower().strip()
    if b == "small_cnn":
        return SmallCNN(num_classes)
    if b == "resnet18":
        return build_resnet18(num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown backbone: {backbone!r}. Use 'resnet18' or 'small_cnn'.")


def build_optimizer(model: nn.Module, backbone: str) -> torch.optim.Optimizer:
    """Build optimizer with lower LR on pretrained ResNet trunk; higher on the head."""
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
    path: Union[str, Path],
    *,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Tuple[nn.Module, dict]:
    """Load `best_model.pt` and rebuild the correct architecture."""
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

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch and return average loss."""
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

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset."""
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
    return y_true, y_pred, y_probs


def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str]
) -> Optional[dict]:
    """Print detailed classification metrics with explanations."""
    n_classes = len(class_names)

    # ----- (1) ACCURACY -----
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 60)
    print("1. ACCURACY")
    print("=" * 60)
    print("Meaning: Of all images, the fraction we predicted correctly.")
    print("Formula: (number correct) / (total number of samples)")
    print(f"Accuracy = {acc:.4f}  ({acc*100:.2f}%)")
    print()

    # ----- (2) PRECISION, RECALL, F1 (per class) -----
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
    
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), average="macro", zero_division=0
    )
    print(f"  Macro avg: Precision = {p_macro:.4f}   Recall = {r_macro:.4f}   F1 = {f_macro:.4f}")
    print()

    # ----- (3) CONFUSION MATRIX -----
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
    if n_classes == 2:
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


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, save_path: str = "roc_curve.png") -> None:
    """Plot and save the ROC curve."""
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

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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


def main() -> None:
    """Main entry point for training."""
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
    print(f"\nFinal classes for training ({len(class_names)} classes):")
    for i, c in enumerate(class_names):
        print(f"  {i}: {c}")

    # Official test set
    possible_test_dirs = [
        data_path / "test",
        data_path / "Test",
        data_dir / "test",
        data_dir / "Test",
        data_dir.parent / "test",
        data_dir.parent / "Test",
    ]

    test_loader, test_class_names = None, None
    for test_dir in possible_test_dirs:
        test_loader, test_class_names = build_test_loader(test_dir, class_names)
        if test_loader is not None:
            print(f"\nTest set found: {test_dir} ({len(test_loader.dataset)} images)")
            print(f"Test set aligned with training classes ({len(test_class_names)} classes)")
            break

    if test_loader is None:
        print("\nNo official test set found; reporting validation metrics only.")
        print("  (This dataset may not have a separate test folder - using validation split instead)")

    num_classes = len(class_names)
    pretrained = args.backbone == "resnet18" and not args.no_pretrained
    model = build_model(args.backbone, num_classes, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.backbone)

    num_epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_acc = -1.0
    checkpoint_path = Path("best_model.pt")

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        # Validation accuracy for checkpointing
        y_true_v, y_pred_v, _ = evaluate(model, val_loader, device, class_names)
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

    print(f"\nBest validation accuracy: {best_val_acc:.4f}  (checkpoint: {checkpoint_path})")

    # Load best checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=True)
    arch = ckpt.get("arch", "small_cnn")
    model = build_model(arch, num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best checkpoint (epoch {ckpt['epoch']}, arch={arch}) for evaluation.")

    # Validation set metrics
    print("\n" + "=" * 60)
    print("VALIDATION SET (20% holdout from train) - best checkpoint")
    print("=" * 60)
    y_true, y_pred, y_probs = evaluate(model, val_loader, device, class_names)
    roc_data = print_metrics(y_true, y_pred, y_probs, class_names)
    if roc_data:
        plot_roc(roc_data["fpr"], roc_data["tpr"], roc_data["auc"], save_path="roc_curve_val.png")

    # Official test set metrics
    if test_loader is not None:
        print("\n" + "=" * 60)
        print("TEST SET (official held-out split) - best checkpoint")
        print("=" * 60)
        y_true_t, y_pred_t, y_probs_t = evaluate(model, test_loader, device, test_class_names)
        print_metrics(y_true_t, y_pred_t, y_probs_t, test_class_names)

    print("\nDone.")


if __name__ == "__main__":
    main()