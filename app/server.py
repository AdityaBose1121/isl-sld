"""
Flask web application server for ISL Sign Language Detection.

Provides a web-based interface for real-time sign language detection
with webcam streaming, sign recognition, and emotion analysis.
"""

import os
import sys
import cv2
import time
import base64
import json
import threading
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from src.inference.realtime_pipeline import RealtimePipeline
from src.utils.config import WEB_APP, EMOTION_EMOJIS


# Global pipeline instance
pipeline = None
camera_thread = None
is_running = False


def create_app():
    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    app.config['SECRET_KEY'] = WEB_APP['secret_key']
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/status')
    def status():
        global pipeline, is_running
        if pipeline:
            state = pipeline.get_state()
            state['is_running'] = is_running
            return jsonify(state)
        return jsonify({'is_running': False, 'sign': '', 'emotion': 'neutral',
                        'sentence': '', 'sentence_history': []})

    @app.route('/api/reset', methods=['POST'])
    def reset():
        global pipeline
        if pipeline:
            pipeline.reset()
        return jsonify({'status': 'ok'})

    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        emit('status', {'connected': True})

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')

    @socketio.on('start_detection')
    def handle_start(data=None):
        global pipeline, camera_thread, is_running

        if is_running:
            emit('status', {'message': 'Already running'})
            return

        camera_id = data.get('camera_id', 0) if data else 0

        try:
            pipeline = RealtimePipeline(camera_id=camera_id)
            is_running = True

            emit('status', {'message': 'Detection started', 'is_running': True})

        except Exception as e:
            emit('error', {'message': f'Failed to start: {str(e)}'})

    @socketio.on('stop_detection')
    def handle_stop():
        global is_running
        is_running = False
        emit('status', {'message': 'Detection stopped', 'is_running': False})

    @socketio.on('frame')
    def handle_frame(data):
        """Process a frame sent from the browser webcam."""
        global pipeline
        if not pipeline or not is_running:
            return

        try:
            # Decode base64 image from browser
            img_data = data.get('image', '')
            if ',' in img_data:
                img_data = img_data.split(',')[1]

            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return

            # Process through pipeline
            annotated, results = pipeline.process_frame(frame)

            # Encode annotated frame back to base64
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Send results back
            emit('detection_result', {
                'frame': f'data:image/jpeg;base64,{frame_b64}',
                'sign': results['sign'],
                'emotion': results['emotion'],
                'emotion_confidence': results['emotion_confidence'],
                'emotion_probs': results['emotion_probs'],
                'sentence': results['sentence'],
                'gloss_buffer': results['gloss_buffer'],
                'sentence_history': [
                    {'sentence': s['sentence'], 'emotion': s['emotion'],
                     'glosses': s['glosses']}
                    for s in results['sentence_history'][-10:]
                ],
                'hands_detected': results['hands_detected'],
                'emoji': EMOTION_EMOJIS.get(results['emotion'], '😐'),
            })

        except Exception as e:
            print(f"Frame processing error: {e}")

    return app, socketio


def camera_loop(socketio):
    """Background camera capture loop (for server-side camera mode)."""
    global pipeline, is_running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        is_running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while is_running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        annotated, results = pipeline.process_frame(frame)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        socketio.emit('detection_result', {
            'frame': f'data:image/jpeg;base64,{frame_b64}',
            'sign': results['sign'],
            'emotion': results['emotion'],
            'emotion_confidence': results['emotion_confidence'],
            'emotion_probs': results['emotion_probs'],
            'sentence': results['sentence'],
            'gloss_buffer': results['gloss_buffer'],
            'sentence_history': [
                {'sentence': s['sentence'], 'emotion': s['emotion'],
                 'glosses': s['glosses']}
                for s in results['sentence_history'][-10:]
            ],
            'hands_detected': results['hands_detected'],
            'emoji': EMOTION_EMOJIS.get(results['emotion'], '😐'),
        })

        time.sleep(1 / 30)  # ~30 FPS

    cap.release()
    is_running = False


if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
