"""
Facial Emotion Recognition CNN.

A lightweight 4-layer CNN trained on FER-2013 to classify facial expressions
into 7 emotion categories. Used to provide sentiment context for ISL recognition.

In ISL, facial expressions serve as grammatical markers:
    - Raised eyebrows → question
    - Furrowed brows → negation
    - Smile → positive sentiment
    - etc.
"""

import torch
import torch.nn as nn
from src.utils.config import EMOTION_MODEL, EMOTION_LABELS


class EmotionCNN(nn.Module):
    """
    Lightweight CNN for facial emotion recognition.

    Architecture:
        Input (1, 48, 48) → 4 Conv blocks → AdaptiveAvgPool → FC → 7 classes
    
    Each conv block: Conv2d → BatchNorm → ReLU → MaxPool
    """

    def __init__(self, num_classes=None, dropout=None):
        super().__init__()
        num_classes = num_classes or EMOTION_MODEL["num_classes"]
        dropout = dropout or EMOTION_MODEL["dropout"]

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 1 → 32 channels, 48×48 → 24×24
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2: 32 → 64 channels, 24×24 → 12×12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3: 64 → 128 channels, 12×12 → 6×6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 4: 128 → 256 channels, 6×6 → global
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 1, 48, 48) — grayscale face images

        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        """
        Predict emotion for a single face image.

        Args:
            x: Tensor of shape (1, 1, 48, 48)

        Returns:
            emotion: string (e.g., 'happy')
            confidence: float
            all_probs: dict mapping emotion → probability
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)

            emotion = EMOTION_LABELS[predicted.item()]
            all_probs = {
                EMOTION_LABELS[i]: probs[0, i].item()
                for i in range(len(EMOTION_LABELS))
            }

        return emotion, confidence.item(), all_probs

    def get_num_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_emotion_model(device=None):
    """Factory function to create and move model to device."""
    from src.utils.config import DEVICE
    device = device or DEVICE
    model = EmotionCNN()
    model = model.to(device)
    total, trainable = model.get_num_parameters()
    print(f"Emotion CNN: {total:,} total params, {trainable:,} trainable")
    return model
