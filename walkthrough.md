# ISL SignVision — Dataset Integration Walkthrough

## What Was Done

All three datasets have been successfully integrated into the project pipeline.

---

## Dataset Summary

| Dataset | Type | Classes | Samples | Status |
|:--------|:-----|:--------|:--------|:-------|
| **FER-2013** | Face images (48x48 grayscale) | 7 emotions | 28,709 train / 7,178 test | Ready to train (has train/test split) |
| **INCLUDE** | ISL sign videos (.MOV) | 249 word signs | 4,043 videos across 15 categories | Needs landmark extraction first |
| **ISL-CSLTR** | Sentence-level videos + frames | 101 sentences / 131 words | 687 videos, 18,863 frames, 1,036 word images | Needs landmark extraction first |

---

## Key Changes Made

### 1. INCLUDE Dataset — Two-Level Nesting Handling

The INCLUDE dataset has a **nested structure** that our original flat parser couldn't handle:

```
include/
    Adjectives/           <-- Category level
        1. loud/          <-- Sign level (with number prefix)
            MVI_5177.MOV  <-- Videos
            MVI_5178.MOV
        83. big large/
            ...
    Animals/
        8. Animal/
            ...
```

**Solution:** Added `process_include_dataset()` in `landmark_extractor.py` that:
- Walks both category and sign levels
- Extracts clean sign names: `"83. big large"` → `"big_large"`
- Flattens to a single-level output: `landmarks/big_large/MVI_xxxx.npy`
- Verified: **249 unique sign classes, 0 duplicates, all with videos**

### 2. ISL-CSLTR Dataset — Sentence-Level Support

Added `process_csltr_videos()` for the sentence-level videos and a new `ISLCSLTRDataset` class:
- Folder names ARE the sentence labels: `"i am very happy"` → `i_am_very_happy`
- Longer max sequence (120 frames vs 30 for word-level)
- 101 sentence classes, 687 videos

### 3. Config Fixes

- Fixed `ISL_CSLTR_DIR` path: `isl_csltr` → `isl-csltr` (matching actual folder name)
- Added `CSLTR_LANDMARKS_DIR` for sentence-level landmarks output

### 4. Train Entry Point

Updated `train.py` with `--dataset` flag:

```bash
python train.py --extract_landmarks                    # Extract both datasets
python train.py --extract_landmarks --dataset include  # INCLUDE only
python train.py --extract_landmarks --dataset csltr    # ISL-CSLTR only
```

---

## Next Steps

### 1. Extract Landmarks (required before training)

> [!IMPORTANT]
> This is a **long-running process** (~2-4 hours for 4,043 INCLUDE videos on CPU).
> MediaPipe runs on CPU for landmark extraction. Each video takes ~1-3 seconds.

```bash
# Extract INCLUDE word-level landmarks
python train.py --extract_landmarks --dataset include

# Extract ISL-CSLTR sentence landmarks (optional, for future expansion)
python train.py --extract_landmarks --dataset csltr
```

### 2. Train Models

```bash
# Train sign recognizer (on GPU, ~2-3 hours with RTX 5070)
python train.py --model sign

# Train emotion CNN (on GPU, ~15 minutes)
python train.py --model emotion
```

### 3. Launch Web App

```bash
python run.py --web
```

---

## Files Modified

| File | Change |
|:-----|:-------|
| `src/data/landmark_extractor.py` | Added `process_include_dataset()` and `process_csltr_videos()` |
| `src/data/dataset.py` | Added `ISLCSLTRDataset` class and `create_train_val_split()` helper |
| `src/utils/config.py` | Fixed `ISL_CSLTR_DIR` path, added `CSLTR_LANDMARKS_DIR` |
| `train.py` | Added `--dataset` flag, updated to use correct extraction methods |
