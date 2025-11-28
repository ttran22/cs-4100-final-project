"""
Train a ResNet18-based emotion classifier on 224x224 face crops in faces/train and faces/test.
Uses Apple Silicon GPU (MPS) if available, otherwise CUDA, otherwise CPU.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from resnet_model import ResNetEmotion

#AI was used throughout the code. 
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

# -------- Device selection (M1/M2 GPU > CUDA > CPU) --------
if torch.backends.mps.is_available():
    device = torch.device("mps")      # Apple Silicon GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")     # NVIDIA GPU
else:
    device = torch.device("cpu")      # Fallback
print("Using device:", device)

FACES_TRAIN_DIR = os.path.join("faces", "train")
FACES_TEST_DIR = os.path.join("faces", "test")

EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 15  # early stopping


def build_dataloaders():
    """
    Build train/test DataLoaders for faces/ using ImageFolder.
    """

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomGrayscale(p=0.1),   # <- NEW
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(FACES_TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(FACES_TEST_DIR, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("ImageFolder class_to_idx:", train_dataset.class_to_idx)
    return train_loader, test_loader, train_dataset.class_to_idx

def mixup_data(x, y, alpha=0.2):
    """Apply MixUp to a batch."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate CutMix bounding box."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2

def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix to a batch."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    shuffled_x = x[index]
    y_a, y_b = y, y[index]

    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x[:, :, y1:y2, x1:x2] = shuffled_x[:, :, y1:y2, x1:x2]

    # Adjust lambda to exact area ratio
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam 

def train_resnet():
    if not os.path.isdir(FACES_TRAIN_DIR) or not os.path.isdir(FACES_TEST_DIR):
        raise FileNotFoundError(
            "faces/train or faces/test not found. Run preprocess_faces.py first."
        )

    train_loader, test_loader, class_to_idx = build_dataloaders()
    num_classes = len(class_to_idx)

    model = ResNetEmotion(num_classes=num_classes, pretrained=True).to(device)

    # Compute class weights
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    counts = np.bincount(all_labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = torch.FloatTensor(inv / inv.sum() * num_classes).to(device)

    # Extra emphasis on fear and sad
    fear_idx = class_to_idx["fear"]
    sad_idx  = class_to_idx["sad"]
    weights[fear_idx] *= 1.3
    weights[sad_idx]  *= 1.3

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("\n=== Training ResNet18 Emotion Model ===")
    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        MIXUP_PROB = 0.3
        CUTMIX_PROB = 0.3

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            r = np.random.rand()
            use_mixup = r < MIXUP_PROB
            use_cutmix = (not use_mixup) and (r < MIXUP_PROB + CUTMIX_PROB)

            if use_mixup:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
            elif use_cutmix:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            else:
                targets_a, targets_b, lam = labels, labels, 1.0

            optimizer.zero_grad()
            outputs = model(images)

            if use_mixup or use_cutmix:
                loss = (lam * criterion(outputs, targets_a) +
                        (1 - lam) * criterion(outputs, targets_b))
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(test_loader)
        val_acc = 100.0 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "resnet_emotion_best.pth")
            print(f"  -> New best model saved ({best_val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")

    # ---- Plot training history ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(train_accs, label="Train")
    ax2.plot(val_accs, label="Val")
    ax2.set_title("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("resnet_emotion_training_history.png", dpi=150)
    plt.close()

    # ---- Confusion matrix on test set ----
    model.load_state_dict(torch.load("resnet_emotion_best.pth", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    label_names = [idx_to_class[i] for i in range(num_classes)]

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("resnet_emotion_confusion_matrix.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    train_resnet()