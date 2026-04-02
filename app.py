"""
SpoiledOrNot Web Interface
Run: python app.py
Then open http://localhost:5000 in your browser
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from collections import deque

from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from baseline import load_model_from_checkpoint, get_val_transform, build_model, get_train_transform
from baseline import build_dataloaders, find_image_folders, get_data_path, train_epoch, evaluate

app = Flask(__name__)

# Global state
camera = None
camera_lock = threading.Lock()
model = None
model_lock = threading.Lock()
fruit_detector = None
transform = None
device = None
class_names = None

# Detection results (thread-safe)
latest_result = {
    'class_name': None,
    'confidence': 0.0,
    'fruit_found': False,
    'timestamp': time.time()
}
result_lock = threading.Lock()

# Training status
training_status = {
    'is_training': False,
    'epoch': 0,
    'total_epochs': 0,
    'train_loss': 0.0,
    'val_acc': 0.0,
    'best_val_acc': 0.0,
    'message': 'Not training'
}
training_lock = threading.Lock()

# Frame buffer for streaming
frame_buffer = deque(maxlen=2)

# Background classification thread
classifier_thread = None
classifier_running = False
current_frame_for_classify = None
classify_lock = threading.Lock()
CLASSIFY_INTERVAL = 1.0  # seconds between classifications


# ── Configuration ──────────────────────────────────────────────────────────────

FRUIT_COCO_IDS = {52, 53, 55, 57}  # banana, apple, orange, carrot
FRUIT_DETECT_THRESHOLD = 0.5


class FruitDetector:
    """Lightweight wrapper around torchvision's Faster R-CNN."""

    def __init__(self, device, threshold=FRUIT_DETECT_THRESHOLD):
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        )
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self._model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        self._model.eval().to(device)
        self._transform = weights.transforms()
        self._device = device
        self._threshold = threshold

    @torch.no_grad()
    def contains_fruit(self, bgr_frame: np.ndarray) -> tuple[bool, list]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self._device)
        preds = self._model(tensor)[0]

        boxes = []
        for label, score, box in zip(preds["labels"], preds["scores"], preds["boxes"]):
            if score >= self._threshold and label.item() in FRUIT_COCO_IDS:
                boxes.append(box.cpu().numpy().astype(int).tolist())

        return len(boxes) > 0, boxes


def load_model_safe(model_path='best_model.pt'):
    """Load the model checkpoint."""
    global model, transform, device, class_names, fruit_detector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path)

    if not model_path.exists():
        return False, f"Model not found: {model_path}. Train first with: python baseline.py"

    try:
        with model_lock:
            model, ckpt = load_model_from_checkpoint(model_path, map_location=device)
            model = model.to(device)
            class_names = ckpt["class_names"]
            transform = get_val_transform(image_size=224)

        # Load fruit detector
        try:
            fruit_detector = FruitDetector(device)
        except Exception as e:
            print(f"Warning: fruit detector failed to load ({e})")
            fruit_detector = None

        return True, f"Model loaded: {ckpt.get('arch', 'unknown')}, classes: {class_names}"
    except Exception as e:
        return False, f"Error loading model: {e}"


def classify_worker():
    """Background thread for classification - doesn't block video stream."""
    global latest_result, current_frame_for_classify

    last_classify = 0

    while classifier_running:
        # Get the latest frame
        frame = None
        with classify_lock:
            if current_frame_for_classify is not None:
                frame = current_frame_for_classify.copy()

        if frame is not None and time.time() - last_classify >= CLASSIFY_INTERVAL:
            if model is not None and transform is not None:
                h, w = frame.shape[:2]
                box_size = min(h, w) * 3 // 4
                x1, y1 = (w - box_size) // 2, (h - box_size) // 2
                x2, y2 = x1 + box_size, y1 + box_size
                crop = frame[y1:y2, x1:x2]

                # Detect fruit
                if fruit_detector is not None:
                    fruit_found, boxes = fruit_detector.contains_fruit(frame)
                else:
                    fruit_found, boxes = True, []

                if fruit_found and crop.size > 0:
                    # Classify
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)

                    try:
                        with model_lock:
                            model.eval()
                            with torch.no_grad():
                                t = transform(pil).unsqueeze(0).to(device)
                                logits = model(t)
                                probs = torch.softmax(logits, dim=1)
                                conf, pred = torch.max(probs, dim=1)

                        class_name = class_names[pred.item()]
                        confidence = conf.item()
                    except Exception as e:
                        class_name = None
                        confidence = 0.0
                else:
                    class_name = None
                    confidence = 0.0

                with result_lock:
                    latest_result = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'fruit_found': fruit_found,
                        'timestamp': time.time()
                    }

            last_classify = time.time()

        time.sleep(0.1)


def draw_overlay(frame, result, frozen=False):
    """Draw UI overlay on frame."""
    h, w = frame.shape[:2]

    # Border color
    if frozen:
        border_color = (255, 180, 0)
    elif not result['fruit_found']:
        border_color = (80, 80, 80)
    elif result['class_name'] and "fresh" in result['class_name'].lower():
        border_color = (0, 220, 0)
    elif result['class_name']:
        border_color = (0, 0, 220)
    else:
        border_color = (100, 100, 100)

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 4)

    # Guide box
    box_size = min(h, w) * 3 // 4
    x1, y1 = (w - box_size) // 2, (h - box_size) // 2
    x2, y2 = x1 + box_size, y1 + box_size
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # Status text
    status = "FROZEN" if frozen else "LIVE"
    color = (0, 200, 255) if frozen else (0, 255, 0)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Result text
    if not result['fruit_found']:
        text = "No fruit detected"
        color = (140, 140, 140)
    elif result['class_name']:
        text = f"{result['class_name'].upper()}: {result['confidence']*100:.1f}%"
        color = (0, 255, 0) if "fresh" in result['class_name'].lower() else (0, 0, 255)
    else:
        text = "Analyzing..."
        color = (180, 180, 180)

    cv2.putText(frame, text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame


def generate_frames(ip_camera=None, camera_index=0, width=640, height=480, rotate=0):
    """Video streaming generator - optimized for low latency."""
    global camera, classifier_running, classifier_thread, current_frame_for_classify

    print(f"[Generate Frames] Starting with dimensions {width}x{height}, rotation={rotate}")

    # Open camera
    if ip_camera:
        print(f"[Generate Frames] Opening IP camera: {ip_camera}")
        cap = cv2.VideoCapture(ip_camera)
    else:
        print(f"[Generate Frames] Opening local camera index: {camera_index}")
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[Generate Frames] Failed to open camera")
        return

    # Get actual camera dimensions
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Generate Frames] Camera native resolution: {actual_width}x{actual_height}")

    # Start background classifier
    classifier_running = True
    classifier_thread = threading.Thread(target=classify_worker, daemon=True)
    classifier_thread.start()

    last_frame_time = 0
    frame_interval = 1.0 / 30  # Target 30 FPS for smooth video
    frozen = False
    frozen_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            # Frame rate limiting - don't encode faster than 30 FPS
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                continue
            last_frame_time = current_time

            # First, rotate if needed
            if rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Then resize to target dimensions
            # After rotation, the frame has new dimensions
            frame = cv2.resize(frame, (width, height))

            # Feed frame to classifier (non-blocking)
            with classify_lock:
                current_frame_for_classify = frame.copy()

            # Get latest result
            with result_lock:
                result = latest_result.copy()

            # Draw overlay
            display = draw_overlay(frame.copy(), result, frozen)

            # Encode with lower quality for faster streaming (quality=70)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
            ret, buffer = cv2.imencode('.jpg', display, encode_params)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        classifier_running = False
        cap.release()


# ── Flask Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Home page with training and camera UI."""
    model_exists = Path('best_model.pt').exists()
    return render_template('index.html', model_exists=model_exists)


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    ip_camera = request.args.get('ip_camera')
    camera_index = int(request.args.get('camera', 0))
    width = int(request.args.get('width', 640))
    height = int(request.args.get('height', 480))
    rotate = int(request.args.get('rotate', 0))

    print(f"[Video Feed] ip_camera={ip_camera}, camera_index={camera_index}, "
          f"width={width}, height={height}, rotate={rotate}")

    return Response(
        generate_frames(ip_camera=ip_camera, camera_index=camera_index, width=width, height=height, rotate=rotate),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/result')
def get_result():
    """Get latest classification result."""
    with result_lock:
        return jsonify(latest_result)


@app.route('/api/training_status')
def get_training_status():
    """Get current training status."""
    with training_lock:
        return jsonify(training_status)


@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start model training in background."""
    global training_status

    with training_lock:
        if training_status['is_training']:
            return jsonify({'success': False, 'message': 'Training already in progress'})

    data = request.json or {}
    backbone = data.get('backbone', 'resnet18')
    epochs = data.get('epochs', 8)

    def train_model():
        global training_status, model, transform, class_names

        with training_lock:
            training_status['is_training'] = True
            training_status['total_epochs'] = epochs
            training_status['message'] = 'Starting training...'

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Get data
            with training_lock:
                training_status['message'] = 'Loading dataset...'

            data_path = get_data_path()
            data_dir, err = find_image_folders(data_path)
            if err:
                data_dir = data_path

            train_loader, val_loader, classes = build_dataloaders(data_dir, device=device)

            with training_lock:
                training_status['message'] = f'Training with {len(classes)} classes...'

            # Build model
            num_classes = len(classes)
            m = build_model(backbone, num_classes, pretrained=True).to(device)
            criterion = torch.nn.CrossEntropyLoss()

            # Optimizer
            if backbone == 'resnet18':
                backbone_params = []
                head_params = []
                for name, p in m.named_parameters():
                    if not p.requires_grad:
                        continue
                    if name.startswith("fc."):
                        head_params.append(p)
                    else:
                        backbone_params.append(p)
                optimizer = torch.optim.Adam([
                    {"params": backbone_params, "lr": 1e-4},
                    {"params": head_params, "lr": 1e-3},
                ])
            else:
                optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            best_val_acc = -1.0

            for epoch in range(1, epochs + 1):
                with training_lock:
                    training_status['epoch'] = epoch
                    training_status['message'] = f'Training epoch {epoch}/{epochs}...'

                loss = train_epoch(m, train_loader, criterion, optimizer, device)
                scheduler.step()

                # Validation
                y_true_v, y_pred_v, _, _ = evaluate(m, val_loader, device, classes)
                from sklearn.metrics import accuracy_score
                val_acc = float(accuracy_score(y_true_v, y_pred_v))

                with training_lock:
                    training_status['train_loss'] = loss
                    training_status['val_acc'] = val_acc
                    training_status['best_val_acc'] = max(training_status['best_val_acc'], val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        "model_state_dict": m.state_dict(),
                        "class_names": classes,
                        "epoch": epoch,
                        "val_accuracy": val_acc,
                        "arch": backbone,
                    }, "best_model.pt")

            # Reload model
            with model_lock:
                model = m
                transform = get_val_transform(image_size=224)
                class_names = classes

            with training_lock:
                training_status['is_training'] = False
                training_status['message'] = f'Training complete! Best accuracy: {best_val_acc:.2%}'

        except Exception as e:
            with training_lock:
                training_status['is_training'] = False
                training_status['message'] = f'Training failed: {str(e)}'

    threading.Thread(target=train_model, daemon=True).start()
    return jsonify({'success': True, 'message': 'Training started'})


@app.route('/api/load_model', methods=['POST'])
def load_model_endpoint():
    """Load the trained model."""
    success, message = load_model_safe('best_model.pt')
    return jsonify({'success': success, 'message': message})


@app.route('/api/config', methods=['POST'])
def set_config():
    """Set app configuration."""
    data = request.json
    if 'rotate' in data:
        app.config['ROTATE'] = data['rotate']
    return jsonify({'success': True})


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpoiledOrNot Web Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Create templates directory
    Path('templates').mkdir(exist_ok=True)

    # Load model if available
    if Path('best_model.pt').exists():
        success, msg = load_model_safe('best_model.pt')
        print(msg)

    print(f"\nStarting SpoiledOrNot Web Interface")
    print(f"Open your browser and go to: http://localhost:{args.port}")
    print(f"Or on your phone: http://<your-computer-ip>:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
