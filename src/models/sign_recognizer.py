"""
Sign Language Recognizer — Transformer Encoder with CTC Loss.

Architecture:
    Input (batch, seq_len, 225) → Linear Embedding (256) → Positional Encoding
    → Transformer Encoder (4 layers, 8 heads) → Linear Head → CTC Loss

This model processes temporal sequences of body landmarks to recognize
ISL word-level signs based on hand and body movements.
"""

import math
import torch
import torch.nn as nn
from src.utils.config import SIGN_MODEL


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SignRecognizer(nn.Module):
    """
    Transformer-based sign language recognizer.

    Processes temporal landmark sequences and outputs sign class predictions.
    Trained with CTC loss for alignment-free recognition.
    """

    def __init__(self, num_classes=None, input_features=None, d_model=None,
                 nhead=None, num_layers=None, d_ff=None, dropout=None, max_seq_len=None):
        super().__init__()

        # Use config defaults if not specified
        self.num_classes = num_classes or SIGN_MODEL["num_classes"]
        input_features = input_features or SIGN_MODEL["input_features"]
        d_model = d_model or SIGN_MODEL["d_model"]
        nhead = nhead or SIGN_MODEL["nhead"]
        num_layers = num_layers or SIGN_MODEL["num_encoder_layers"]
        d_ff = d_ff or SIGN_MODEL["d_feedforward"]
        dropout = dropout or SIGN_MODEL["dropout"]
        max_seq_len = max_seq_len or SIGN_MODEL["max_seq_len"]

        self.d_model = d_model

        # Input projection: (225 → d_model)
        self.input_projection = nn.Sequential(
            nn.Linear(input_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Classification head (for CTC: output at every timestep)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.num_classes + 1),  # +1 for CTC blank token
        )

        # Also provide a pooled classification head (for standard CE loss)
        self.pooled_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, return_ctc=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_features)
            return_ctc: If True, return per-timestep logits for CTC loss.
                        If False, return pooled classification logits.

        Returns:
            If return_ctc=True:  (batch, seq_len, num_classes+1) — CTC logits
            If return_ctc=False: (batch, num_classes) — classification logits
        """
        # Project input features to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        if return_ctc:
            # Per-timestep classification for CTC
            logits = self.classifier(x)  # (batch, seq_len, num_classes+1)
            return logits
        else:
            # Global average pooling → classification
            pooled = x.mean(dim=1)  # (batch, d_model)
            logits = self.pooled_classifier(pooled)  # (batch, num_classes)
            return logits

    def predict(self, x):
        """
        Predict the sign class for an input sequence.

        Args:
            x: Input tensor of shape (1, seq_len, input_features)

        Returns:
            predicted_class: int
            confidence: float
            top_k: list of (class_idx, confidence) tuples
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, return_ctc=False)
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)

            # Top-5 predictions
            top5_probs, top5_indices = probs.topk(5, dim=-1)
            top_k = list(zip(
                top5_indices[0].cpu().numpy().tolist(),
                top5_probs[0].cpu().numpy().tolist()
            ))

        return predicted.item(), confidence.item(), top_k

    def get_num_parameters(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_sign_model(num_classes=None, device=None):
    """Factory function to create and move model to device."""
    from src.utils.config import DEVICE
    device = device or DEVICE
    model = SignRecognizer(num_classes=num_classes)
    model = model.to(device)
    total, trainable = model.get_num_parameters()
    print(f"Sign Recognizer: {total:,} total params, {trainable:,} trainable")
    return model
