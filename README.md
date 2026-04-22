# 🤟 ISL SignVision — Indian Sign Language Detection

Advanced real-time Indian Sign Language (ISL) recognition system that combines **hand/body movement recognition** with **facial sentiment analysis** to produce meaningful, context-aware English sentences.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

## 🏗️ Architecture

```
Webcam → MediaPipe Holistic → [Hand+Pose Landmarks] → Transformer Encoder (CTC) → Sign Glosses ─┐
                             → [Face ROI]           → Emotion CNN (FER-2013)   → Emotion Label ──┤
                                                                                                  ├→ Sentence Former → "I am happy today" 😊
```

### Three-Model Pipeline

| Model | Architecture | Purpose | Training Data |
|:------|:-------------|:--------|:-------------|
| **Sign Recognizer** | Transformer Encoder + CTC | Word-level ISL sign recognition | INCLUDE (263 signs) |
| **Emotion Analyzer** | 4-layer CNN | Facial expression detection | FER-2013 (7 emotions) |
| **Sentence Former** | Rule-based + Gemini API | Gloss→sentence conversion with ISL grammar | N/A |

## 📦 Datasets Required

### 1. INCLUDE Dataset (Sign Recognition)
- **263 ISL word signs**, 4,287 videos
- Download: [http://bit.ly/include_dl](http://bit.ly/include_dl)
- Extract to: `data/include/`

### 2. FER-2013 Dataset (Emotion Recognition)
- **35,887 face images**, 7 emotion classes
- Download: [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Extract to: `data/fer2013/` (with `train/` and `test/` subdirectories)

### 3. ISL-CSLTR (Optional — Sentence-Level)
- Download: [Kaggle ISL-CSLTR](https://www.kaggle.com/datasets/kartiksaxena/islcsltr-indian-sign-language-dataset)
- Extract to: `data/isl_csltr/`

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download & Prepare Datasets

Download the datasets above, then extract landmarks:

```bash
python train.py --extract_landmarks
```

### 3. Train Models

```bash
# Train both models
python train.py --model all

# Or train individually
python train.py --model sign --epochs 100
python train.py --model emotion --epochs 50
```

### 4. Run Inference

```bash
# Webcam mode (OpenCV window)
python run.py

# Web application mode
python run.py --web
# Then open http://localhost:5000
```

## 🖥️ Web Application

The web app provides a premium dark-themed interface with:
- 📹 Live webcam feed with landmark overlay
- ✋ Real-time sign recognition display
- 😊 Facial emotion detection with confidence chart
- 💬 Automatic sentence formation from sign sequences
- 📜 Sentence history

## 📁 Project Structure

```
sign-language-detection2/
├── data/                       # Datasets (download separately)
├── models/                     # Saved model weights
├── src/
│   ├── data/
│   │   ├── landmark_extractor.py   # MediaPipe landmark extraction
│   │   ├── dataset.py              # PyTorch Dataset classes
│   │   └── preprocessing.py       # Normalization & augmentation
│   ├── models/
│   │   ├── sign_recognizer.py     # Transformer Encoder + CTC
│   │   ├── emotion_cnn.py         # Facial emotion CNN
│   │   └── sentence_former.py     # Gloss → sentence conversion
│   ├── training/
│   │   ├── train_sign.py          # Sign model training
│   │   └── train_emotion.py       # Emotion model training
│   ├── inference/
│   │   └── realtime_pipeline.py   # Real-time webcam pipeline
│   └── utils/
│       ├── config.py              # Hyperparameters & settings
│       └── visualization.py       # Drawing utilities
├── app/
│   ├── server.py                  # Flask + SocketIO backend
│   ├── templates/index.html       # Web UI
│   └── static/                    # CSS & JS
├── train.py                       # Training entry point
├── run.py                         # Inference entry point
└── requirements.txt
```

## ⚙️ Configuration

All hyperparameters are in `src/utils/config.py`:
- Model architectures (d_model, heads, layers)
- Training settings (batch size, LR, epochs)
- Inference settings (FPS, confidence thresholds)
- Gemini API key (optional, via `GEMINI_API_KEY` env var)

## 🧠 How It Works

### Sign Recognition
1. **MediaPipe Holistic** extracts 75 body landmarks (33 pose + 21×2 hands) per frame
2. Landmarks are normalized relative to body center and shoulder width
3. 30-frame sequences are fed into a **Transformer Encoder**
4. CTC loss handles temporal alignment without frame-level labels

### Facial Sentiment
- Face ROI is extracted and classified into 7 emotions
- In ISL, facial expressions are **grammatical markers** (raised eyebrows = question)
- Emotion context enriches the sentence formation

### Sentence Formation
- ISL uses **SOV** (Subject-Object-Verb) order → converted to English **SVO**
- Emotion modifies tone (e.g., sad face + "I feel" → "I am feeling sad")
- Optional: Gemini API for complex sentence construction

## 📊 Expected Performance

| Metric | Target |
|:-------|:-------|
| Sign recognition accuracy (top-5) | >70% |
| Emotion recognition accuracy | >65% |
| Real-time inference FPS | >15 FPS |

## 📄 License

MIT License
