"""
Main training entry point.

Usage:
    python train.py --extract_landmarks                       # Extract landmarks (8 workers)
    python train.py --extract_landmarks --workers 10          # Extract with 10 parallel workers
    python train.py --extract_landmarks --dataset include     # Extract INCLUDE only
    python train.py --extract_landmarks --dataset csltr       # Extract ISL-CSLTR only
    python train.py --extract_landmarks --workers 1           # Sequential (single-threaded)
    python train.py --model sign                              # Train sign recognizer
    python train.py --model emotion                           # Train emotion CNN
    python train.py --model all                               # Train both models
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def extract_landmarks(dataset="all", num_workers=8):
    """Extract landmarks from video datasets into .npy files.
    
    Uses multiprocessing for parallel extraction across CPU cores.
    MediaPipe runs on CPU only (no GPU support on Windows), but
    with multiple workers it's ~8× faster.
    """
    from src.utils.config import INCLUDE_DIR, ISL_CSLTR_DIR, LANDMARKS_DIR, CSLTR_LANDMARKS_DIR

    if num_workers > 1:
        # Parallel mode — uses module-level functions (each worker creates its own extractor)
        from src.data.landmark_extractor import process_include_parallel, process_csltr_parallel

        if dataset in ("include", "all"):
            print("=" * 60)
            print("Extracting landmarks: INCLUDE dataset (PARALLEL)")
            print(f"  Input:   {INCLUDE_DIR}")
            print(f"  Output:  {LANDMARKS_DIR}")
            print(f"  Workers: {num_workers}")
            print("=" * 60)

            if not os.path.exists(INCLUDE_DIR):
                print("ERROR: INCLUDE dataset not found at " + INCLUDE_DIR)
                print("Download from: http://bit.ly/include_dl")
            else:
                process_include_parallel(INCLUDE_DIR, LANDMARKS_DIR,
                                         num_workers=num_workers)
            print()

        if dataset in ("csltr", "all"):
            print("=" * 60)
            print("Extracting landmarks: ISL-CSLTR dataset (PARALLEL)")
            print(f"  Input:   {ISL_CSLTR_DIR}")
            print(f"  Output:  {CSLTR_LANDMARKS_DIR}")
            print(f"  Workers: {num_workers}")
            print("=" * 60)

            if not os.path.exists(ISL_CSLTR_DIR):
                print("ERROR: ISL-CSLTR dataset not found at " + ISL_CSLTR_DIR)
                print("Download from: https://www.kaggle.com/datasets/kartiksaxena/islcsltr-indian-sign-language-dataset")
            else:
                process_csltr_parallel(ISL_CSLTR_DIR, CSLTR_LANDMARKS_DIR,
                                       num_workers=num_workers)
            print()
    else:
        # Sequential mode — single extractor instance
        from src.data.landmark_extractor import LandmarkExtractor

        extractor = LandmarkExtractor(static_image_mode=True)
        try:
            if dataset in ("include", "all"):
                print("=" * 60)
                print("Extracting landmarks: INCLUDE dataset")
                print("  Input:  " + INCLUDE_DIR)
                print("  Output: " + LANDMARKS_DIR)
                print("=" * 60)

                if not os.path.exists(INCLUDE_DIR):
                    print("ERROR: INCLUDE dataset not found at " + INCLUDE_DIR)
                    print("Download from: http://bit.ly/include_dl")
                else:
                    extractor.process_include_dataset(INCLUDE_DIR, LANDMARKS_DIR)
                print()

            if dataset in ("csltr", "all"):
                print("=" * 60)
                print("Extracting landmarks: ISL-CSLTR dataset")
                print("  Input:  " + ISL_CSLTR_DIR)
                print("  Output: " + CSLTR_LANDMARKS_DIR)
                print("=" * 60)

                if not os.path.exists(ISL_CSLTR_DIR):
                    print("ERROR: ISL-CSLTR dataset not found at " + ISL_CSLTR_DIR)
                    print("Download from: https://www.kaggle.com/datasets/kartiksaxena/islcsltr-indian-sign-language-dataset")
                else:
                    extractor.process_csltr_videos(ISL_CSLTR_DIR, CSLTR_LANDMARKS_DIR)
                print()
        finally:
            extractor.close()



def main():
    parser = argparse.ArgumentParser(description="ISL Sign Language Detection - Training")
    parser.add_argument("--model", type=str, choices=["sign", "emotion", "all"],
                        default="all", help="Which model to train")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--extract_landmarks", action="store_true",
                        help="Extract landmarks from video dataset first")
    parser.add_argument("--dataset", type=str, choices=["include", "csltr", "all"],
                        default="all",
                        help="Which dataset to extract landmarks from (used with --extract_landmarks)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers for landmark extraction (default: 8)")
    args = parser.parse_args()

    # Step 1: Extract landmarks if requested
    if args.extract_landmarks:
        extract_landmarks(args.dataset, num_workers=args.workers)

    # Step 2: Train models
    if args.model in ["sign", "all"]:
        print("=" * 60)
        print("Training: Sign Language Recognizer")
        print("=" * 60)
        from src.training.train_sign import train_sign_model
        from src.utils.config import LANDMARKS_DIR

        train_sign_model(
            data_dir=args.data_dir or LANDMARKS_DIR,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        print()

    if args.model in ["emotion", "all"]:
        print("=" * 60)
        print("Training: Facial Emotion Recognition CNN")
        print("=" * 60)
        from src.training.train_emotion import train_emotion_model
        from src.utils.config import FER_DIR

        train_emotion_model(
            data_dir=args.data_dir or FER_DIR,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        print()

    print("Training complete!")


if __name__ == "__main__":
    main()
