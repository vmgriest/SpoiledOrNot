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
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

from baseline import (
    build_dataloaders,
    build_model,
    evaluate,
    find_image_folders,
    get_data_path,
    get_train_transform,
    get_val_transform,
    load_model_from_checkpoint,
    train_epoch,
)

app = Flask(__name__)

# Global state
camera: Optional[cv2.VideoCapture] = None
camera_lock = threading.Lock()
model: Optional[torch.nn.Module] = None
model_lock = threading.Lock()
fruit_detector: Optional[Any] = None
transform: Optional[transforms.Compose] = None
device: Optional[torch.device] = None
class_names: Optional[list] = None

# Detection results (thread-safe)
latest_result: Dict[str, Any] = {
    "class_name": None,
    "confidence": 0.0,
    "fruit_found": False,
    "timestamp": time.time(),
}
result_lock = threading.Lock()

# Training status
training_status: Dict[str, Any] = {
    "is_training": False,
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_acc": 0.0,
    "best_val_acc": 0.0,
    "message": "Not training",
}
training_lock = threading.Lock()

# Frame buffer for streaming
frame_buffer: deque = deque(maxlen=2)

# Background classification thread
classifier_thread: Optional[threading.Thread] = None
classifier_running = False
current_frame_for_classify: Optional[np.ndarray] = None
classify_lock = threading.Lock()
CLASSIFY_INTERVAL = 2.0  # classify every 2 seconds to reduce lag
last_classify_time = 0.0


# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# COCO class IDs for produce detection (only general fruit/vegetable classes)
# We don't use COCO for specific produce type identification - we use the freshness model's
# class names to determine if it's an Apple, Tomato, Banana, etc.
# This avoids confusion like tomatoes being detected as apples.
PRODUCE_COCO_IDS = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57}
# 52: banana, 53: apple, 55: orange, 57: carrot
# 46-51, 54, 56: other food items that might be produce-like
PRODUCE_DETECT_THRESHOLD = 0.4
FRUIT_DETECT_THRESHOLD = 0.4  # Kept for backward compatibility


class FruitDetector:
    """Lightweight wrapper around torchvision's Faster R-CNN for produce detection."""

    def __init__(self, device: torch.device, threshold: float = FRUIT_DETECT_THRESHOLD):
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self._model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        self._model.eval().to(device)
        self._transform = weights.transforms()
        self._device = device
        self._threshold = threshold

    @torch.no_grad()
    def contains_fruit(self, bgr_frame: np.ndarray) -> Tuple[bool, list]:
        """Check if frame contains produce. Returns (found, bounding_boxes)."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self._device)
        preds = self._model(tensor)[0]

        boxes = []
        for label, score, box in zip(preds["labels"], preds["scores"], preds["boxes"]):
            label_id = label.item()
            if label_id in PRODUCE_COCO_IDS and score >= PRODUCE_DETECT_THRESHOLD:
                boxes.append(box.cpu().numpy().astype(int).tolist())

        return len(boxes) > 0, boxes


def load_model_safe(model_path: str = "best_model.pt") -> Tuple[bool, str]:
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


def normalize_freshness_class(raw_class: str, class_names: list, probs: list = None) -> Tuple[str, str]:
    """
    Normalize class name to Fresh/Rotten regardless of produce type.
    Also extracts the produce type (Apple, Tomato, Banana, etc.) from the class name.
    Handles formats like: 'Apple_Fresh', 'fresh_apple', 'Fresh', 'Rotten Banana', etc.

    Returns: (freshness_status, produce_type)
    """
    raw_lower = raw_class.lower()
    produce_type = "Unknown"

    # Extract produce type from class name patterns like "Apple_Fresh", "Tomato_Rotten"
    # Common produce types we're looking for
    known_produce = ["potato", "apple", "tomato", "orange", "banana", "okra", "cucumber"]

    for produce in known_produce:
        if produce in raw_lower:
            produce_type = produce.capitalize()
            break

    # If no known produce found, try to extract from before _fresh or _rotten
    if produce_type == "Unknown" and "_" in raw_class:
        # Pattern like "Apple_Fresh" or "Tomato_Rotten"
        parts = raw_class.lower().split("_")
        if len(parts) >= 2:
            # The produce type is usually the first part
            candidate = parts[0].capitalize()
            # Exclude freshness indicators
            if candidate.lower() not in ["fresh", "rotten", "stale", "spoiled"]:
                produce_type = candidate

    # Check if this is a freshness-based class name
    has_fresh = "fresh" in raw_lower
    has_rotten = any(word in raw_lower for word in ["rotten", "stale", "spoiled", "bad"])

    if has_fresh and not has_rotten:
        return "Fresh", produce_type
    elif has_rotten and not has_fresh:
        return "Rotten", produce_type
    elif has_fresh and has_rotten:
        return raw_class.replace("_", " ").title(), produce_type

    # For binary classification, try to infer from class names
    if len(class_names) == 2:
        fresh_idx = -1
        rotten_idx = -1
        for i, name in enumerate(class_names):
            name_lower = name.lower()
            if "fresh" in name_lower:
                fresh_idx = i
            elif any(word in name_lower for word in ["rotten", "stale", "spoiled"]):
                rotten_idx = i

        if probs is not None and fresh_idx >= 0 and rotten_idx >= 0:
            freshness = "Fresh" if probs[fresh_idx] > probs[rotten_idx] else "Rotten"
            return freshness, produce_type

    return raw_class.replace("_", " ").title(), produce_type


def classify_worker() -> None:
    """Background thread for classification - does not block video stream."""
    global latest_result, current_frame_for_classify

    last_classify = 0

    while classifier_running:
        # Get the latest frame
        frame: Optional[np.ndarray] = None
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

                        raw_class = class_names[pred.item()]
                        # Normalize to Fresh/Rotten for consistent display and extract produce type
                        class_name, produce_type = normalize_freshness_class(raw_class, class_names, probs[0].cpu().numpy())
                        confidence = conf.item()
                    except Exception:
                        class_name = None
                        confidence = 0.0
                        produce_type = "Unknown"
                else:
                    class_name = None
                    confidence = 0.0
                    produce_type = "Unknown"

                with result_lock:
                    latest_result = {
                        "class_name": class_name,
                        "confidence": confidence,
                        "fruit_found": fruit_found,
                        "produce_type": produce_type,
                        "timestamp": time.time(),
                    }

            last_classify = time.time()

        time.sleep(0.1)


def draw_overlay(frame: np.ndarray, result: Dict[str, Any], frozen: bool = False) -> np.ndarray:
    """Draw UI overlay on frame."""
    h, w = frame.shape[:2]

    # Border color - check for "fresh" in normalized class name
    class_name = result.get("class_name", "")
    is_fresh = class_name and ("fresh" in class_name.lower() or class_name.lower() == "fresh")

    if frozen:
        border_color = (255, 180, 0)
    elif not result["fruit_found"]:
        border_color = (80, 80, 80)
    elif class_name and is_fresh:
        border_color = (0, 220, 0)
    elif class_name:
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
    produce_type = result.get("produce_type", "Unknown")
    if not result["fruit_found"]:
        text = "No produce detected"
        color = (140, 140, 140)
    elif class_name:
        text = f"{class_name.upper()}: {result['confidence']*100:.1f}%"
        color = (0, 255, 0) if is_fresh else (0, 0, 255)
        # Show produce type above the freshness result
        if produce_type and produce_type != "Unknown":
            type_text = f"Type: {produce_type}"
            cv2.putText(frame, type_text, (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    else:
        text = "Analyzing..."
        color = (180, 180, 180)

    cv2.putText(frame, text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame


def generate_frames(
    ip_camera: Optional[str] = None,
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    rotate: int = 0
):
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
    frozen_frame: Optional[np.ndarray] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            # Frame rate limiting - do not encode faster than 30 FPS
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
            ret, buffer = cv2.imencode(".jpg", display, encode_params)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

    finally:
        classifier_running = False
        cap.release()


# ------------------------------------------------------------------------------
# FLASK ROUTES
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    """Home page with training and camera UI."""
    model_exists = Path("best_model.pt").exists()
    return render_template("index.html", model_exists=model_exists)


@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    ip_camera = request.args.get("ip_camera")
    camera_index = int(request.args.get("camera", 0))
    width = int(request.args.get("width", 640))
    height = int(request.args.get("height", 480))
    rotate = int(request.args.get("rotate", 0))

    print(
        f"[Video Feed] ip_camera={ip_camera}, camera_index={camera_index}, "
        f"width={width}, height={height}, rotate={rotate}"
    )

    return Response(
        generate_frames(
            ip_camera=ip_camera,
            camera_index=camera_index,
            width=width,
            height=height,
            rotate=rotate,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/result")
def get_result():
    """Get latest classification result."""
    with result_lock:
        return jsonify(latest_result)


@app.route("/api/training_status")
def get_training_status():
    """Get current training status."""
    with training_lock:
        return jsonify(training_status)


@app.route("/api/start_training", methods=["POST"])
def start_training():
    """Start model training in background."""
    global training_status

    with training_lock:
        if training_status["is_training"]:
            return jsonify({"success": False, "message": "Training already in progress"})

    data = request.json or {}
    backbone = data.get("backbone", "resnet18")
    epochs = data.get("epochs", 8)

    def train_model():
        global training_status, model, transform, class_names

        with training_lock:
            training_status["is_training"] = True
            training_status["total_epochs"] = epochs
            training_status["message"] = "Starting training..."

        try:
            device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Get data
            with training_lock:
                training_status["message"] = "Loading dataset..."

            data_path = get_data_path()
            data_dir, err = find_image_folders(data_path)
            if err:
                data_dir = data_path

            train_loader, val_loader, classes = build_dataloaders(data_dir, device=device_local)

            with training_lock:
                training_status["message"] = f"Training with {len(classes)} classes..."

            # Build model
            num_classes = len(classes)
            m = build_model(backbone, num_classes, pretrained=True).to(device_local)
            criterion = torch.nn.CrossEntropyLoss()

            # Optimizer
            if backbone == "resnet18":
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
                    training_status["epoch"] = epoch
                    training_status["message"] = f"Training epoch {epoch}/{epochs}..."

                loss = train_epoch(m, train_loader, criterion, optimizer, device_local)
                scheduler.step()

                # Validation
                y_true_v, y_pred_v, _, _ = evaluate(m, val_loader, device_local, classes)
                from sklearn.metrics import accuracy_score
                val_acc = float(accuracy_score(y_true_v, y_pred_v))

                with training_lock:
                    training_status["train_loss"] = loss
                    training_status["val_acc"] = val_acc
                    training_status["best_val_acc"] = max(training_status["best_val_acc"], val_acc)

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
                training_status["is_training"] = False
                training_status["message"] = f"Training complete! Best accuracy: {best_val_acc:.2%}"

        except Exception as e:
            with training_lock:
                training_status["is_training"] = False
                training_status["message"] = f"Training failed: {str(e)}"

    threading.Thread(target=train_model, daemon=True).start()
    return jsonify({"success": True, "message": "Training started"})


@app.route("/api/load_model", methods=["POST"])
def load_model_endpoint():
    """Load the trained model."""
    success, message = load_model_safe("best_model.pt")
    return jsonify({"success": success, "message": message})


@app.route("/api/config", methods=["POST"])
def set_config():
    """Set app configuration."""
    data = request.json
    if data and "rotate" in data:
        app.config["ROTATE"] = data["rotate"]
    return jsonify({"success": True})


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpoiledOrNot Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Create templates directory
    Path("templates").mkdir(exist_ok=True)

    # Load model if available
    if Path("best_model.pt").exists():
        success, msg = load_model_safe("best_model.pt")
        print(msg)

    print("\nStarting SpoiledOrNot Web Interface")
    print(f"Open your browser and go to: http://localhost:{args.port}")
    print(f"Or on your phone: http://<your-computer-ip>:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
