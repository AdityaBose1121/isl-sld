"""
Visualization utilities for landmark drawing and UI overlays.
"""

import cv2
import numpy as np
from src.utils.config import EMOTION_EMOJIS, EMOTION_LABELS


def draw_info_panel(frame, sign_text, emotion, emotion_conf, sentence, fps):
    """
    Draw an information panel overlay on the video frame.

    Args:
        frame: BGR image
        sign_text: Currently recognized sign
        emotion: Detected emotion string
        emotion_conf: Emotion confidence (0-1)
        sentence: Current accumulated sentence
        fps: Current FPS
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent dark panel at the bottom
    panel_h = 140
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # FPS counter (top-left)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)

    y_base = h - panel_h + 30

    # Current sign
    if sign_text:
        cv2.putText(frame, f"Sign: {sign_text}", (15, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Emotion with confidence bar
    if emotion:
        emoji = EMOTION_EMOJIS.get(emotion, "")
        cv2.putText(frame, f"Emotion: {emotion} {emoji}", (15, y_base + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)

        # Confidence bar
        bar_x = 280
        bar_w = 150
        bar_h = 15
        cv2.rectangle(frame, (bar_x, y_base + 22), (bar_x + bar_w, y_base + 22 + bar_h),
                      (60, 60, 60), -1)
        fill_w = int(bar_w * emotion_conf)
        color = (0, 255, 0) if emotion_conf > 0.7 else (0, 255, 255) if emotion_conf > 0.4 else (0, 100, 255)
        cv2.rectangle(frame, (bar_x, y_base + 22), (bar_x + fill_w, y_base + 22 + bar_h),
                      color, -1)
        cv2.putText(frame, f"{emotion_conf:.0%}", (bar_x + bar_w + 10, y_base + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Sentence
    if sentence:
        # Truncate if too long
        max_chars = w // 12
        display_sentence = sentence if len(sentence) <= max_chars else sentence[-max_chars:]
        cv2.putText(frame, f">> {display_sentence}", (15, y_base + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    return frame


def draw_hand_status(frame, left_detected, right_detected):
    """Draw hand detection status indicators."""
    h, w = frame.shape[:2]
    y = 60

    # Left hand
    color = (0, 255, 0) if left_detected else (80, 80, 80)
    cv2.putText(frame, "L", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.circle(frame, (35, y - 7), 5, color, -1)

    # Right hand
    color = (0, 255, 0) if right_detected else (80, 80, 80)
    cv2.putText(frame, "R", (55, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.circle(frame, (80, y - 7), 5, color, -1)

    return frame


def create_emotion_chart(all_probs, width=300, height=200):
    """
    Create a bar chart visualization of emotion probabilities.

    Args:
        all_probs: dict mapping emotion → probability
        width: Chart width
        height: Chart height

    Returns:
        chart: BGR image of the chart
    """
    chart = np.zeros((height, width, 3), dtype=np.uint8)
    chart[:] = (30, 30, 40)

    if not all_probs:
        return chart

    bar_height = 20
    spacing = 5
    max_bar_width = width - 120
    y_offset = 10

    colors = {
        "angry": (60, 60, 255),
        "disgust": (60, 180, 60),
        "fear": (200, 150, 60),
        "happy": (60, 255, 255),
        "sad": (255, 150, 60),
        "surprise": (255, 60, 255),
        "neutral": (180, 180, 180),
    }

    for i, emotion in enumerate(EMOTION_LABELS):
        prob = all_probs.get(emotion, 0)
        y = y_offset + i * (bar_height + spacing)

        # Label
        cv2.putText(chart, emotion[:7], (5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Bar background
        bar_x = 75
        cv2.rectangle(chart, (bar_x, y), (bar_x + max_bar_width, y + bar_height),
                      (50, 50, 60), -1)

        # Bar fill
        fill_w = int(max_bar_width * prob)
        color = colors.get(emotion, (180, 180, 180))
        cv2.rectangle(chart, (bar_x, y), (bar_x + fill_w, y + bar_height), color, -1)

        # Percentage
        cv2.putText(chart, f"{prob:.0%}", (bar_x + max_bar_width + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return chart
