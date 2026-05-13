"""Fine-tune ResNet50 for AI-image detection on the MS COCOAI dataset.

Dataset layout expected at ~/datasets/ms_cocoai (HuggingFace disk format):
    columns: image (PIL / filepath), label (int: 0=real, 1=AI)

Training schedule:
    Epochs 1-5  : only the final FC layer is trainable, lr=1e-4
    Epochs 6-10 : all layers unfrozen, lr=1e-5

Best validation-accuracy checkpoint saved to training/best_model.pt.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from datasets import load_from_disk
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────

DATASET_PATH = Path.home() / "datasets" / "ms_cocoai"
CHECKPOINT_PATH = Path(__file__).parent / "best_model.pt"

# ── Hyper-parameters ──────────────────────────────────────────────────────────

BATCH_SIZE = 32
NUM_WORKERS = 0
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 5
PHASE1_LR = 1e-4
PHASE2_LR = 1e-5

# ── Transforms ────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class CocoAIDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        img = sample["Image"]
        if isinstance(img, str):
            img = Image.open(img)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        label = int(sample["Label_A"])
        return self.transform(img), label


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def set_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze or unfreeze every layer except the final FC."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = not frozen


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch. Returns (avg_loss, accuracy)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and split dataset
    print(f"Loading dataset from {DATASET_PATH} ...")
    hf_ds = load_from_disk(str(DATASET_PATH))

    train_hf = hf_ds["train"]
    val_hf = hf_ds["validation"]
    print(f"Train: {len(train_hf):,}  Val: {len(val_hf):,}")

    train_ds = CocoAIDataset(train_hf, train_transform)
    val_ds = CocoAIDataset(val_hf, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    header = f"{'Epoch':>6}  {'Phase':<8}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Best':>5}"
    print("\n" + header)
    print("-" * len(header))

    for phase, epochs, lr, frozen in [
        ("frozen", PHASE1_EPOCHS, PHASE1_LR, True),
        ("full",   PHASE2_EPOCHS, PHASE2_LR, False),
    ]:
        set_frozen(model, frozen)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[{phase}]  lr={lr}  trainable params={trainable:,}")

        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
        )
        start_epoch = 1 if phase == "frozen" else PHASE1_EPOCHS + 1

        for epoch in range(start_epoch, start_epoch + epochs):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                torch.save(model.state_dict(), CHECKPOINT_PATH)

            print(
                f"{epoch:>6}  {phase:<8}  {train_loss:>10.4f}  {train_acc:>8.2%}"
                f"  {val_loss:>8.4f}  {val_acc:>6.2%}  {'✓' if is_best else '':>5}"
            )

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2%}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
