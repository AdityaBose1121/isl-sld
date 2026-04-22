"""
Real-time Sign Language Detection Pipeline.

Combines webcam capture, landmark extraction, sign recognition,
emotion analysis, and sentence formation into a unified real-time system.
"""

import cv2
import time
import numpy as np
import torch
from collections import deque

from src.data.landmark_extractor import LandmarkExtractor
from src.data.preprocessing import LandmarkNormalizer, pad_or_truncate_sequence
from src.models.sign_recognizer import SignRecognizer
from src.models.emotion_cnn import EmotionCNN
from src.models.sentence_former import SentenceFormer, build_sentence_former
from src.utils.config import (
    DEVICE, INFERENCE, SIGN_MODEL_PATH, EMOTION_MODEL_PATH,
    EMOTION_LABELS, EMOTION_EMOJIS, SIGN_MODEL
)
from src.utils.visualization import draw_info_panel, draw_hand_status, create_emotion_chart


class RealtimePipeline:
    """
    Real-time ISL sign language detection pipeline.

    Pipeline:
        1. Capture webcam frame
        2. Extract landmarks (MediaPipe Holistic)
        3. Buffer N frames of landmarks
        4. Every N frames → run sign recognition
        5. Run emotion detection on face ROI
        6. Accumulate glosses → form sentence on pause
    """

    def __init__(self, sign_model_path=None, emotion_model_path=None,
                 camera_id=None, device=None):
        self.device = device or DEVICE
        self.camera_id = camera_id if camera_id is not None else INFERENCE["camera_id"]

        # Frame buffer for temporal sign recognition
        self.frame_buffer_size = INFERENCE["frame_buffer_size"]
        self.landmark_buffer = deque(maxlen=self.frame_buffer_size)

        # Sentence accumulation
        self.gloss_buffer = []
        self.last_sign_time = time.time()
        self.sentence_pause_threshold = INFERENCE["sentence_pause_threshold"]
        self.confidence_threshold = INFERENCE["confidence_threshold"]

        # Emotion smoothing
        self.emotion_history = deque(maxlen=INFERENCE["emotion_smoothing_window"])

        # Results
        self.current_sign = ""
        self.current_emotion = "neutral"
        self.current_emotion_conf = 0.0
        self.current_emotion_probs = {}
        self.current_sentence = ""
        self.sentence_history = []

        # Initialize components
        self.extractor = LandmarkExtractor()
        self.normalizer = LandmarkNormalizer()
        self.sentence_former = build_sentence_former()

        self.class_names = []
        
        # Load models
        self.sign_model = self._load_sign_model(sign_model_path or SIGN_MODEL_PATH)
        self.emotion_model = self._load_emotion_model(emotion_model_path or EMOTION_MODEL_PATH)

    def _load_sign_model(self, model_path):
        """Load the trained sign recognition model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            num_classes = checkpoint.get('num_classes', SIGN_MODEL["num_classes"])
            self.class_names = checkpoint.get('class_names', [])

            model = SignRecognizer(num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            print(f"[OK] Sign model loaded ({num_classes} classes, "
                  f"val_acc: {checkpoint.get('val_acc', 'N/A')}%)")
            return model
        except FileNotFoundError:
            print(f"[!] Sign model not found at {model_path}")
            print("  The system will run in demo mode with random predictions.")
            print("  Train the model first: python train.py --model sign")
            return None
        except Exception as e:
            print(f"[!] Error loading sign model: {e}")
            return None

    def _load_emotion_model(self, model_path):
        """Load the trained emotion recognition model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model = EmotionCNN()
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            print(f"[OK] Emotion model loaded (val_acc: {checkpoint.get('val_acc', 'N/A')}%)")
            return model
        except FileNotFoundError:
            print(f"[!] Emotion model not found at {model_path}")
            print("  Emotion detection will be disabled.")
            return None
        except Exception as e:
            print(f"[!] Error loading emotion model: {e}")
            return None

    def process_frame(self, frame):
        """
        Process a single frame through the full pipeline.

        Args:
            frame: BGR image from webcam

        Returns:
            annotated_frame: Frame with visualizations
            results: dict with current predictions
        """
        # 1. Extract landmarks and face ROI
        landmarks, face_roi, mp_results = self.extractor.extract_landmarks(frame)

        # 2. Buffer landmarks
        self.landmark_buffer.append(landmarks)

        # 3. Detect hands
        left_detected = mp_results.left_hand_landmarks is not None
        right_detected = mp_results.right_hand_landmarks is not None
        hands_detected = left_detected or right_detected

        # 4. Run sign recognition when buffer is full
        if len(self.landmark_buffer) == self.frame_buffer_size and hands_detected:
            self._recognize_sign()

        # 5. Run emotion recognition
        if face_roi is not None:
            self._recognize_emotion(face_roi)

        # 6. Check for sentence boundary (pause in signing)
        if hands_detected:
            self.last_sign_time = time.time()
        elif time.time() - self.last_sign_time > self.sentence_pause_threshold:
            if self.gloss_buffer:
                self._form_sentence()

        # 7. Draw visualizations
        annotated = self.extractor.draw_landmarks(frame.copy(), mp_results)
        annotated = draw_hand_status(annotated, left_detected, right_detected)

        results = {
            "sign": self.current_sign,
            "emotion": self.current_emotion,
            "emotion_confidence": self.current_emotion_conf,
            "emotion_probs": self.current_emotion_probs,
            "sentence": self.current_sentence,
            "gloss_buffer": list(self.gloss_buffer),
            "sentence_history": list(self.sentence_history),
            "hands_detected": hands_detected,
        }

        return annotated, results

    def _recognize_sign(self):
        """Run sign recognition on the buffered landmarks."""
        if self.sign_model is None:
            return

        # Convert buffer to numpy array
        seq = np.array(list(self.landmark_buffer), dtype=np.float32)

        # Normalize
        seq = self.normalizer.normalize_sequence(seq)

        # Pad/truncate to expected length
        seq = pad_or_truncate_sequence(seq, self.frame_buffer_size)

        # Convert to tensor
        tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)

        # Predict
        predicted_class, confidence, top_k = self.sign_model.predict(tensor)

        if confidence >= self.confidence_threshold:
            if self.class_names and predicted_class < len(self.class_names):
                sign_name = self.class_names[predicted_class]
            else:
                sign_name = f"sign_{predicted_class}"

            self.current_sign = sign_name
            self.gloss_buffer.append(sign_name)
            self.last_sign_time = time.time()

            # Clear buffer for next sign
            self.landmark_buffer.clear()

    def _recognize_emotion(self, face_roi):
        """Run emotion recognition on the face ROI."""
        if self.emotion_model is None:
            return

        # Prepare tensor
        face_tensor = torch.FloatTensor(face_roi.astype(np.float32) / 255.0)
        face_tensor = face_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # Predict
        emotion, confidence, all_probs = self.emotion_model.predict(face_tensor)

        # Smooth emotion predictions
        self.emotion_history.append(emotion)

        # Use most common emotion in history
        if self.emotion_history:
            from collections import Counter
            emotion_counts = Counter(self.emotion_history)
            self.current_emotion = emotion_counts.most_common(1)[0][0]

        self.current_emotion_conf = confidence
        self.current_emotion_probs = all_probs

    def _form_sentence(self):
        """Form a sentence from accumulated glosses + emotion."""
        if not self.gloss_buffer:
            return

        sentence, metadata = self.sentence_former.form_sentence(
            self.gloss_buffer,
            emotion=self.current_emotion,
            emotion_confidence=self.current_emotion_conf
        )

        if sentence:
            self.current_sentence = sentence
            self.sentence_history.append({
                "sentence": sentence,
                "glosses": list(self.gloss_buffer),
                "emotion": self.current_emotion,
                "method": metadata.get("method", "unknown"),
                "timestamp": time.time(),
            })

        # Clear gloss buffer for next sentence
        self.gloss_buffer.clear()

    def run_webcam(self, show_window=True):
        """
        Run the pipeline with webcam input.

        Args:
            show_window: Whether to show the OpenCV window
        """
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {self.camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n" + "=" * 60)
        print("ISL Sign Language Detection — LIVE")
        print("=" * 60)
        print("Controls:")
        print("  q — Quit")
        print("  r — Reset sentence buffer")
        print("  s — Save current sentence")
        print("  c — Toggle emotion chart")
        print("=" * 60 + "\n")

        fps_counter = deque(maxlen=30)
        show_emotion_chart = False

        while cap.isOpened():
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Process frame
            annotated, results = self.process_frame(frame)

            # Calculate FPS
            elapsed = time.time() - start
            fps_counter.append(1.0 / max(elapsed, 1e-6))
            fps = np.mean(fps_counter)

            # Draw info panel
            annotated = draw_info_panel(
                annotated,
                sign_text=results["sign"],
                emotion=results["emotion"],
                emotion_conf=results["emotion_confidence"],
                sentence=results["sentence"],
                fps=fps
            )

            # Draw emotion chart if enabled
            if show_emotion_chart and results["emotion_probs"]:
                chart = create_emotion_chart(results["emotion_probs"])
                ch, cw = chart.shape[:2]
                x_offset = annotated.shape[1] - cw - 10
                y_offset = 10
                annotated[y_offset:y_offset+ch, x_offset:x_offset+cw] = chart

            if show_window:
                cv2.imshow("ISL Sign Language Detection", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.gloss_buffer.clear()
                    self.current_sign = ""
                    self.current_sentence = ""
                    print("Buffer reset")
                elif key == ord('s'):
                    if self.current_sentence:
                        print(f"Saved: {self.current_sentence}")
                elif key == ord('c'):
                    show_emotion_chart = not show_emotion_chart

        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        self.extractor.close()

        print("\nSession complete!")
        if self.sentence_history:
            print("\nSentence History:")
            for i, entry in enumerate(self.sentence_history, 1):
                emoji = EMOTION_EMOJIS.get(entry["emotion"], "")
                print(f"  {i}. {entry['sentence']} {emoji}")
                print(f"     Glosses: {' → '.join(entry['glosses'])}")

    def get_state(self):
        """Get current pipeline state (for web API)."""
        return {
            "sign": self.current_sign,
            "emotion": self.current_emotion,
            "emotion_confidence": self.current_emotion_conf,
            "emotion_probs": self.current_emotion_probs,
            "sentence": self.current_sentence,
            "gloss_buffer": list(self.gloss_buffer),
            "sentence_history": [
                {
                    "sentence": s["sentence"],
                    "emotion": s["emotion"],
                    "glosses": s["glosses"],
                }
                for s in self.sentence_history[-10:]
            ],
        }

    def reset(self):
        """Reset all buffers and state."""
        self.landmark_buffer.clear()
        self.gloss_buffer.clear()
        self.emotion_history.clear()
        self.current_sign = ""
        self.current_emotion = "neutral"
        self.current_emotion_conf = 0.0
        self.current_sentence = ""
        self.sentence_history.clear()
