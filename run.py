"""
Main inference entry point.

Usage:
    python run.py              # Run webcam inference
    python run.py --web        # Run web application
    python run.py --camera 1   # Use camera index 1
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="ISL Sign Language Detection — Inference")
    parser.add_argument("--web", action="store_true", help="Launch web application")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--sign_model", type=str, default=None, help="Path to sign model")
    parser.add_argument("--emotion_model", type=str, default=None, help="Path to emotion model")
    args = parser.parse_args()

    if args.web:
        # Launch Flask web app
        print("Starting web application...")
        from app.server import create_app
        app, socketio = create_app()
        socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
    else:
        # Run webcam inference directly
        from src.inference.realtime_pipeline import RealtimePipeline

        pipeline = RealtimePipeline(
            sign_model_path=args.sign_model,
            emotion_model_path=args.emotion_model,
            camera_id=args.camera,
        )
        pipeline.run_webcam(show_window=True)


if __name__ == "__main__":
    main()
