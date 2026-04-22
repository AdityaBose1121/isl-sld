"""
Training script for the Facial Emotion Recognition CNN (FER-2013).

Usage:
    python -m src.training.train_emotion --data_dir data/fer2013 --epochs 50
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FERDataset
from src.models.emotion_cnn import EmotionCNN, build_emotion_model
from src.utils.config import EMOTION_TRAINING, DEVICE, MODELS_DIR, FER_DIR, EMOTION_LABELS


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(dim=-1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'acc': f'{100.0 * correct / total:.1f}%'
        })

    return total_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    per_class_correct = [0] * len(EMOTION_LABELS)
    per_class_total = [0] * len(EMOTION_LABELS)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(dim=-1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            for i in range(labels.size(0)):
                label = labels[i].item()
                per_class_total[label] += 1
                if predicted[i] == label:
                    per_class_correct[label] += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    per_class_acc = {}
    for i, name in enumerate(EMOTION_LABELS):
        if per_class_total[i] > 0:
            per_class_acc[name] = 100.0 * per_class_correct[i] / per_class_total[i]
        else:
            per_class_acc[name] = 0.0

    return avg_loss, accuracy, per_class_acc


def train_emotion_model(data_dir=None, epochs=None, batch_size=None, lr=None, device=None):
    """Full training pipeline for the emotion recognition CNN."""
    data_dir = data_dir or FER_DIR
    epochs = epochs or EMOTION_TRAINING["epochs"]
    batch_size = batch_size or EMOTION_TRAINING["batch_size"]
    lr = lr or EMOTION_TRAINING["learning_rate"]
    device = device or DEVICE

    print(f"{'=' * 60}")
    print(f"Facial Emotion Recognition Training")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"{'=' * 60}")

    # Load datasets
    print("\nLoading FER-2013 dataset...")
    train_dataset = FERDataset(data_dir, split="train", augment=True)
    test_dataset = FERDataset(data_dir, split="test", augment=False)

    if len(train_dataset) == 0:
        print("ERROR: No training data found. Please download FER-2013 from:")
        print("  https://www.kaggle.com/datasets/msambare/fer2013")
        print(f"  and extract to: {data_dir}")
        return None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=min(4, os.cpu_count() or 1), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=min(4, os.cpu_count() or 1), pin_memory=True)

    # Build model
    model = build_emotion_model(device=device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=EMOTION_TRAINING["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=EMOTION_TRAINING["step_size"],
        gamma=EMOTION_TRAINING["gamma"]
    )

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "emotion_cnn.pth")

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, per_class = validate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}% | "
              f"Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'per_class_acc': per_class,
            }, save_path)
            print(f"  [OK] Best model saved (val_acc: {val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= EMOTION_TRAINING["early_stopping_patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final report
    print(f"\n{'=' * 60}")
    print(f"Training complete! Best validation accuracy: {best_val_acc:.1f}%")
    print(f"\nPer-class accuracy:")
    checkpoint = torch.load(save_path, map_location=device)
    for emotion, acc in checkpoint['per_class_acc'].items():
        print(f"  {emotion:12s}: {acc:.1f}%")
    print(f"\nModel saved to: {save_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotion CNN")
    parser.add_argument("--data_dir", type=str, default=FER_DIR)
    parser.add_argument("--epochs", type=int, default=EMOTION_TRAINING["epochs"])
    parser.add_argument("--batch_size", type=int, default=EMOTION_TRAINING["batch_size"])
    parser.add_argument("--lr", type=float, default=EMOTION_TRAINING["learning_rate"])
    args = parser.parse_args()

    train_emotion_model(
        data_dir=args.data_dir, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
    )
