"""
Training script for the Sign Language Recognizer (Transformer + CTC/CE).

Usage:
    python -m src.training.train_sign --data_dir data/landmarks --epochs 100
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

from src.data.dataset import INCLUDEDataset
from src.models.sign_recognizer import SignRecognizer, build_sign_model
from src.utils.config import SIGN_TRAINING, SIGN_MODEL, DEVICE, MODELS_DIR, LANDMARKS_DIR


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch with optional AMP, return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None and device.type == 'cuda'

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (sequences, labels) in enumerate(pbar):
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(sequences, return_ctc=False)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(sequences, return_ctc=False)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(dim=-1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'acc': f'{100.0 * correct / total:.1f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate and return average loss, accuracy, and top-5 accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validating", leave=False):
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences, return_ctc=False)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(dim=-1)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = logits.topk(min(5, logits.size(-1)), dim=-1)
            for i in range(labels.size(0)):
                if labels[i] in top5_pred[i]:
                    correct_top5 += 1

            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    top5_accuracy = 100.0 * correct_top5 / total
    return avg_loss, accuracy, top5_accuracy


def train_sign_model(data_dir=None, epochs=None, batch_size=None, lr=None, device=None):
    """
    Full training pipeline for the sign recognition model.

    Args:
        data_dir: Path to pre-extracted landmarks directory
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: torch device
    """
    data_dir = data_dir or LANDMARKS_DIR
    epochs = epochs or SIGN_TRAINING["epochs"]
    batch_size = batch_size or SIGN_TRAINING["batch_size"]
    lr = lr or SIGN_TRAINING["learning_rate"]
    device = device or DEVICE
    use_amp = SIGN_TRAINING.get("mixed_precision", False) and device.type == "cuda"

    print("=" * 60)
    print("Sign Language Recognizer Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"AMP (Mixed Precision): {use_amp}")
    print(f"Data dir: {data_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    full_dataset = INCLUDEDataset(data_dir, augment=False)

    if len(full_dataset) == 0:
        print("ERROR: No data found. Please download the INCLUDE dataset and")
        print("run landmark extraction first:")
        print("  1. Download: http://bit.ly/include_dl")
        print("  2. Extract landmarks: python extract_landmarks.py")
        return None

    num_classes = full_dataset.get_num_classes()
    print(f"Classes: {num_classes}, Samples: {len(full_dataset)}")

    # Train/val/test split
    val_size = int(len(full_dataset) * SIGN_TRAINING["val_split"])
    test_size = int(len(full_dataset) * SIGN_TRAINING["test_split"])
    train_size = len(full_dataset) - val_size - test_size

    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Enable augmentation for training set
    train_dataset = INCLUDEDataset(data_dir, augment=True)
    train_indices = train_set.indices
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=min(4, os.cpu_count() or 1), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=min(4, os.cpu_count() or 1), pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=min(4, os.cpu_count() or 1), pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Build model
    model = build_sign_model(num_classes=num_classes, device=device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=SIGN_TRAINING["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # AMP GradScaler for RTX GPU
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using Automatic Mixed Precision (FP16) for faster training")

    best_val_acc = 0.0
    patience_counter = 0
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "sign_recognizer.pth")

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler
        )
        val_loss, val_acc, val_top5 = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%, Top5: {val_top5:.1f}% | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_top5': val_top5,
                'num_classes': num_classes,
                'class_names': full_dataset.class_names,
                'config': SIGN_MODEL,
            }, save_path)
            print(f"  [OK] Best model saved (val_acc: {val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= SIGN_TRAINING["early_stopping_patience"]:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for "
                      f"{SIGN_TRAINING['early_stopping_patience']} epochs)")
                break

    # Final test evaluation
    print(f"\n{'=' * 60}")
    print("Final evaluation on test set...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_top5 = validate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.1f}%, Top-5: {test_top5:.1f}%")
    print(f"Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {save_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL Sign Recognizer")
    parser.add_argument("--data_dir", type=str, default=LANDMARKS_DIR)
    parser.add_argument("--epochs", type=int, default=SIGN_TRAINING["epochs"])
    parser.add_argument("--batch_size", type=int, default=SIGN_TRAINING["batch_size"])
    parser.add_argument("--lr", type=float, default=SIGN_TRAINING["learning_rate"])
    args = parser.parse_args()

    train_sign_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
