/**
 * ISL SignVision — Frontend Application Logic
 * 
 * Handles webcam capture, SocketIO communication with the Flask backend,
 * and updates the UI with real-time detection results.
 */

// ============================================================================
// State
// ============================================================================
const state = {
    isRunning: false,
    socket: null,
    webcamStream: null,
    videoEl: null,
    canvasEl: null,
    canvasCtx: null,
    captureInterval: null,
    fps: 0,
    fpsFrames: [],
    lastFrameTime: 0,
};

// Emotion emoji mapping
const EMOTION_EMOJIS = {
    angry: '😠', disgust: '🤢', fear: '😨',
    happy: '😊', sad: '😢', surprise: '😲', neutral: '😐',
};

const EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Cache DOM elements
    state.videoEl = document.getElementById('webcamVideo');
    state.canvasEl = document.getElementById('outputCanvas');
    state.canvasCtx = state.canvasEl.getContext('2d');

    // Initialize Socket.IO
    initSocket();

    // Build emotion chart bars
    buildEmotionChart();

    // Enumerate cameras
    enumerateCameras();

    // Bind events
    document.getElementById('startBtn').addEventListener('click', toggleDetection);
    document.getElementById('resetBtn').addEventListener('click', resetBuffers);
    document.getElementById('clearHistoryBtn').addEventListener('click', clearHistory);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && e.target === document.body) {
            e.preventDefault();
            toggleDetection();
        }
        if (e.code === 'KeyR' && e.target === document.body) {
            resetBuffers();
        }
    });
});

// ============================================================================
// Socket.IO
// ============================================================================
function initSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.socket = io(window.location.origin, {
        transports: ['websocket', 'polling'],
    });

    state.socket.on('connect', () => {
        console.log('Connected to server');
    });

    state.socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateStatus(false);
    });

    state.socket.on('detection_result', handleDetectionResult);

    state.socket.on('error', (data) => {
        console.error('Server error:', data.message);
        showNotification(data.message, 'error');
    });

    state.socket.on('status', (data) => {
        if (data.is_running !== undefined) {
            updateStatus(data.is_running);
        }
    });
}

// ============================================================================
// Webcam
// ============================================================================
async function startWebcam() {
    try {
        const cameraId = document.getElementById('cameraSelect').value;
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user',
            },
            audio: false,
        };

        // Use specific device if selected
        if (cameraId && cameraId !== '0') {
            constraints.video.deviceId = { exact: cameraId };
        }

        state.webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        state.videoEl.srcObject = state.webcamStream;

        // Wait for video to be ready
        await new Promise((resolve) => {
            state.videoEl.onloadedmetadata = () => {
                state.canvasEl.width = state.videoEl.videoWidth;
                state.canvasEl.height = state.videoEl.videoHeight;
                resolve();
            };
        });

        return true;
    } catch (err) {
        console.error('Failed to access webcam:', err);
        showNotification('Failed to access camera. Please allow camera permissions.', 'error');
        return false;
    }
}

function stopWebcam() {
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(track => track.stop());
        state.webcamStream = null;
        state.videoEl.srcObject = null;
    }
}

async function enumerateCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        const select = document.getElementById('cameraSelect');
        select.innerHTML = '';

        videoDevices.forEach((device, idx) => {
            const option = document.createElement('option');
            option.value = device.deviceId || idx.toString();
            option.textContent = device.label || `Camera ${idx + 1}`;
            select.appendChild(option);
        });
    } catch (err) {
        console.log('Could not enumerate cameras');
    }
}

// ============================================================================
// Frame Capture & Sending
// ============================================================================
function startFrameCapture() {
    const CAPTURE_FPS = 15; // Send frames at 15 FPS to server
    const interval = 1000 / CAPTURE_FPS;

    state.captureInterval = setInterval(() => {
        if (!state.isRunning || !state.videoEl.videoWidth) return;

        // Create an offscreen canvas for capture if not exists
        if (!state.captureCanvas) {
            state.captureCanvas = document.createElement('canvas');
            state.captureCtx = state.captureCanvas.getContext('2d');
            state.captureCanvas.width = state.canvasEl.width;
            state.captureCanvas.height = state.canvasEl.height;
        }

        // Draw current video frame to offscreen canvas
        state.captureCtx.drawImage(
            state.videoEl, 0, 0,
            state.captureCanvas.width, state.captureCanvas.height
        );

        // Convert to base64 and send to server
        const frameData = state.captureCanvas.toDataURL('image/jpeg', 0.7);
        state.socket.emit('frame', { image: frameData });

        // Update FPS
        const now = performance.now();
        state.fpsFrames.push(now);
        state.fpsFrames = state.fpsFrames.filter(t => now - t < 1000);
        state.fps = state.fpsFrames.length;
        document.getElementById('fpsCounter').textContent = `${state.fps} FPS`;
    }, interval);
}

function stopFrameCapture() {
    if (state.captureInterval) {
        clearInterval(state.captureInterval);
        state.captureInterval = null;
    }
}

// ============================================================================
// Detection Control
// ============================================================================
async function toggleDetection() {
    if (state.isRunning) {
        // Stop
        stopFrameCapture();
        stopWebcam();
        state.socket.emit('stop_detection');
        updateStatus(false);
    } else {
        // Start
        const success = await startWebcam();
        if (!success) return;

        state.socket.emit('start_detection', {
            camera_id: document.getElementById('cameraSelect').value
        });

        startFrameCapture();
        updateStatus(true);
    }
}

function resetBuffers() {
    fetch('/api/reset', { method: 'POST' });
    updateSignDisplay('');
    updateSentenceDisplay('');
    document.getElementById('glossBufferContent').textContent = '—';
    showNotification('Buffers reset', 'info');
}

function clearHistory() {
    document.getElementById('historyList').innerHTML =
        '<p class="history-empty">No sentences yet</p>';
}

// ============================================================================
// UI Updates
// ============================================================================
function updateStatus(running) {
    state.isRunning = running;
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');
    const btn = document.getElementById('startBtn');
    const resetBtn = document.getElementById('resetBtn');
    const overlay = document.getElementById('videoOverlay');
    const canvas = document.getElementById('outputCanvas');

    if (running) {
        dot.classList.add('active');
        text.textContent = 'Live';
        btn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
            </svg> Stop Detection`;
        btn.classList.add('active');
        resetBtn.disabled = false;
        overlay.classList.add('hidden');
        canvas.classList.add('active');
    } else {
        dot.classList.remove('active');
        text.textContent = 'Offline';
        btn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3"/>
            </svg> Start Detection`;
        btn.classList.remove('active');
        resetBtn.disabled = true;
        overlay.classList.remove('hidden');
        canvas.classList.remove('active');
    }
}

function handleDetectionResult(data) {
    // Update processed frame
    if (data.frame) {
        const img = new Image();
        img.onload = () => {
            state.canvasCtx.drawImage(img, 0, 0, state.canvasEl.width, state.canvasEl.height);
        };
        img.src = data.frame;
        document.getElementById('outputCanvas').classList.add('active');
    }

    // Update sign display
    if (data.sign) {
        updateSignDisplay(data.sign);
    }

    // Update gloss buffer
    if (data.gloss_buffer && data.gloss_buffer.length > 0) {
        document.getElementById('glossBufferContent').textContent =
            data.gloss_buffer.join(' → ');
    }

    // Update emotion
    updateEmotionDisplay(data.emotion, data.emotion_confidence, data.emoji, data.emotion_probs);

    // Update sentence
    if (data.sentence) {
        updateSentenceDisplay(data.sentence);
    }

    // Update history
    if (data.sentence_history && data.sentence_history.length > 0) {
        updateHistory(data.sentence_history);
    }

    // Update hand indicators
    updateHandIndicators(data.hands_detected);
}

function updateSignDisplay(sign) {
    const el = document.getElementById('signDisplay');
    if (sign) {
        el.innerHTML = `<span class="sign-text">${sign.replace(/_/g, ' ')}</span>`;
    } else {
        el.innerHTML = '<span class="sign-placeholder">Waiting...</span>';
    }
}

function updateEmotionDisplay(emotion, confidence, emoji, probs) {
    if (!emotion) return;

    document.getElementById('emotionEmoji').textContent = emoji || EMOTION_EMOJIS[emotion] || '😐';
    document.getElementById('emotionLabel').textContent = emotion;
    document.getElementById('confidenceFill').style.width = `${(confidence || 0) * 100}%`;
    document.getElementById('confidenceText').textContent = `${Math.round((confidence || 0) * 100)}%`;

    // Update emotion chart bars
    if (probs) {
        EMOTIONS.forEach(em => {
            const fill = document.querySelector(`.emotion-bar-fill.${em}`);
            const val = document.querySelector(`.emotion-bar-value[data-emotion="${em}"]`);
            if (fill) {
                const pct = (probs[em] || 0) * 100;
                fill.style.width = `${pct}%`;
            }
            if (val) {
                val.textContent = `${Math.round((probs[em] || 0) * 100)}%`;
            }
        });
    }
}

function updateSentenceDisplay(sentence) {
    const el = document.getElementById('sentenceDisplay');
    if (sentence) {
        el.innerHTML = `<p class="sentence-text">"${sentence}"</p>`;
    } else {
        el.innerHTML = '<p class="sentence-placeholder">Sentences will appear here as you sign...</p>';
    }
}

function updateHistory(history) {
    const list = document.getElementById('historyList');
    const currentCount = list.querySelectorAll('.history-item').length;

    // Only update if there are new entries
    if (history.length <= currentCount) return;

    list.innerHTML = '';
    history.reverse().forEach((entry) => {
        const item = document.createElement('div');
        item.className = 'history-item';
        const emoji = EMOTION_EMOJIS[entry.emotion] || '😐';
        item.innerHTML = `
            <div class="history-sentence">${emoji} "${entry.sentence}"</div>
            <div class="history-meta">
                <span>Glosses: ${entry.glosses ? entry.glosses.join(' → ') : '—'}</span>
                <span>Emotion: ${entry.emotion}</span>
            </div>
        `;
        list.appendChild(item);
    });
}

function updateHandIndicators(detected) {
    const left = document.getElementById('leftHandIndicator');
    const right = document.getElementById('rightHandIndicator');

    if (detected) {
        left.classList.add('detected');
        right.classList.add('detected');
    } else {
        left.classList.remove('detected');
        right.classList.remove('detected');
    }
}

// ============================================================================
// Emotion Chart Builder
// ============================================================================
function buildEmotionChart() {
    const container = document.getElementById('emotionChart');
    container.innerHTML = '';

    EMOTIONS.forEach(em => {
        const row = document.createElement('div');
        row.className = 'emotion-bar-row';
        row.innerHTML = `
            <span class="emotion-bar-label">${em}</span>
            <div class="emotion-bar-track">
                <div class="emotion-bar-fill ${em}" style="width: 0%"></div>
            </div>
            <span class="emotion-bar-value" data-emotion="${em}">0%</span>
        `;
        container.appendChild(row);
    });
}

// ============================================================================
// Notifications
// ============================================================================
function showNotification(message, type = 'info') {
    // Create notification element
    const notif = document.createElement('div');
    notif.style.cssText = `
        position: fixed; top: 20px; right: 20px; z-index: 1000;
        padding: 12px 20px; border-radius: 10px;
        background: ${type === 'error' ? 'rgba(244, 63, 94, 0.9)' : 'rgba(139, 92, 246, 0.9)'};
        color: white; font-size: 0.85rem; font-weight: 500;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        animation: slideIn 0.3s ease-out;
        font-family: var(--font-sans);
    `;
    notif.textContent = message;
    document.body.appendChild(notif);

    setTimeout(() => {
        notif.style.opacity = '0';
        notif.style.transition = 'opacity 0.3s ease';
        setTimeout(() => notif.remove(), 300);
    }, 3000);
}
