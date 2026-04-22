"""
Configuration and hyperparameters for the ISL Sign Language Detection system.
Centralizes all paths, model parameters, and training settings.
"""

import os

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Load .env file automatically (if present) so GEMINI_API_KEY etc. are available
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), ".env")
    load_dotenv(_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed yet; env vars must be set manually

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data directories — raw datasets
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INCLUDE_DIR = os.path.join(DATA_DIR, "include")
FER_DIR = os.path.join(DATA_DIR, "fer2013")
ISL_CSLTR_DIR = os.path.join(DATA_DIR, "isl-csltr")  # hyphen matches actual folder name

# Pre-extracted landmark directories
LANDMARKS_DIR = os.path.join(DATA_DIR, "landmarks")           # INCLUDE word-level landmarks
CSLTR_LANDMARKS_DIR = os.path.join(DATA_DIR, "landmarks_csltr")  # ISL-CSLTR sentence landmarks

# Model save directories
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SIGN_MODEL_PATH = os.path.join(MODELS_DIR, "sign_recognizer.pth")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_cnn.pth")

# ============================================================================
# DEVICE
# ============================================================================
if _HAS_TORCH:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"  # Fallback for environments without PyTorch (e.g. extraction)

# ============================================================================
# MEDIAPIPE LANDMARKS
# ============================================================================
# Landmark counts from MediaPipe Holistic
NUM_POSE_LANDMARKS = 33       # Upper body + legs
NUM_HAND_LANDMARKS = 21       # Per hand
NUM_FACE_LANDMARKS = 468      # Full face mesh

# We use a subset of the most informative landmarks:
# - 33 pose landmarks × 3 (x, y, z) = 99
# - 21 left hand landmarks × 3 = 63
# - 21 right hand landmarks × 3 = 63
# Total = 225 features per frame
NUM_LANDMARK_FEATURES = (NUM_POSE_LANDMARKS + NUM_HAND_LANDMARKS * 2) * 3  # 225

# ============================================================================
# SIGN RECOGNITION MODEL (Transformer Encoder + CTC)
# ============================================================================
SIGN_MODEL = {
    "d_model": 512,              # Larger model: RTX 5070 can handle 512 dim
    "nhead": 8,                  # Number of attention heads
    "num_encoder_layers": 6,     # Deeper: 6 layers with 8GB VRAM
    "d_feedforward": 1024,       # 2× d_model feedforward
    "dropout": 0.1,              # Dropout rate
    "max_seq_len": 30,           # Maximum sequence length (frames)
    "num_classes": 263,          # Max classes (actual count auto-detected during training)
    "input_features": NUM_LANDMARK_FEATURES,  # 225
}

# ============================================================================
# EMOTION CNN MODEL
# ============================================================================
EMOTION_MODEL = {
    "input_size": 48,            # FER-2013 images are 48x48
    "num_classes": 7,            # 7 emotion classes
    "dropout": 0.5,
}

EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
]

# Mapping emotions to emojis for display
EMOTION_EMOJIS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
}

# ============================================================================
# TRAINING — SIGN MODEL
# ============================================================================
SIGN_TRAINING = {
    "batch_size": 64,             # RTX 5070 8GB — larger batch
    "epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "scheduler": "cosine",
    "early_stopping_patience": 15,
    "val_split": 0.15,
    "test_split": 0.10,
    "num_workers": 4,
    "mixed_precision": True,      # AMP for faster training on RTX 5070
}

# ============================================================================
# TRAINING — EMOTION MODEL
# ============================================================================
EMOTION_TRAINING = {
    "batch_size": 128,            # RTX 5070 8GB — larger batch for CNN
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "step",
    "step_size": 15,
    "gamma": 0.1,
    "early_stopping_patience": 10,
    "num_workers": 4,
    "mixed_precision": True,      # AMP for faster training
}

# ============================================================================
# INFERENCE
# ============================================================================
INFERENCE = {
    "camera_id": 0,                   # Default webcam
    "frame_buffer_size": 30,          # Number of frames to buffer for sign recognition
    "fps_target": 30,                 # Target FPS
    "sentence_pause_threshold": 1.5,  # Seconds of no signing → sentence boundary
    "confidence_threshold": 0.5,      # Minimum confidence for sign prediction
    "emotion_smoothing_window": 10,   # Smooth emotion over N frames
}

# ============================================================================
# GEMINI API (Optional — for advanced sentence formation)
# ============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
USE_GEMINI = bool(GEMINI_API_KEY)

# ============================================================================
# ISL SIGN VOCABULARY
# ============================================================================
# Common ISL words from the INCLUDE dataset (subset for demo)
# The full list is loaded from the dataset during training
DEMO_VOCABULARY = [
    "hello", "thank_you", "please", "sorry", "help",
    "water", "food", "home", "school", "work",
    "mother", "father", "friend", "doctor", "teacher",
    "good", "bad", "happy", "sad", "pain",
    "yes", "no", "want", "need", "like",
    "come", "go", "eat", "drink", "sleep",
    "morning", "evening", "today", "tomorrow", "yesterday",
    "name", "what", "where", "when", "how",
    "i", "you", "he", "she", "they",
    "big", "small", "hot", "cold", "new",
]

# ============================================================================
# WEB APP
# ============================================================================
WEB_APP = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "secret_key": "isl-sign-language-detection-secret",
}
