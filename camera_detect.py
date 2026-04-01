"""
Real-time fruit freshness detection using laptop camera.
Classifies continuously on the live feed — no need to press SPACE.
Press SPACE to freeze/unfreeze, ESC or 'q' to quit.

Run: python camera_detect.py
     python camera_detect.py --model best_model.pt
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from baseline import load_model_from_checkpoint, get_val_transform


# ── Configuration ──────────────────────────────────────────────────────────────

# Seconds between each classification (lower = more responsive, higher CPU use)
CLASSIFY_INTERVAL = 1.0

# COCO class IDs that count as "fruit/produce" in torchvision's Faster R-CNN
# 52=banana, 53=apple, 55=orange, 57=carrot — covers common fruits
FRUIT_COCO_IDS = {52, 53, 55, 57}
FRUIT_DETECT_THRESHOLD = 0.5   # min detector confidence to count as fruit


# ── Fruit detector (Faster R-CNN, COCO) ───────────────────────────────────────

class FruitDetector:
    """
    Lightweight wrapper around torchvision's Faster R-CNN (MobileNet backbone).
    Only flags detections whose COCO class is in FRUIT_COCO_IDS.
    No internet download needed after the first run — weights are cached locally.
    """

    def __init__(self, device, threshold=FRUIT_DETECT_THRESHOLD):
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        )
        print("Loading fruit detector (Faster R-CNN MobileNet)…")
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self._model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        self._model.eval().to(device)
        self._transform = weights.transforms()
        self._device = device
        self._threshold = threshold
        print("Fruit detector ready.")

    @torch.no_grad()
    def contains_fruit(self, bgr_frame: np.ndarray) -> tuple[bool, list]:
        """
        Returns (fruit_found: bool, boxes: list of (x1,y1,x2,y2) for found fruits).
        Runs on the full frame so nothing is accidentally cropped out.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self._device)
        preds = self._model(tensor)[0]

        boxes = []
        for label, score, box in zip(preds["labels"], preds["scores"], preds["boxes"]):
            if score >= self._threshold and label.item() in FRUIT_COCO_IDS:
                boxes.append(box.cpu().numpy().astype(int).tolist())

        return len(boxes) > 0, boxes


# ── Text / drawing helpers ─────────────────────────────────────────────────────

def put_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                              scale=0.8, color=(255, 255, 255), thickness=2,
                              bg_color=(0, 0, 0), alpha=0.6):
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - text_h - 10), (x + text_w + 10, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x + 5, y), font, scale, color, thickness)
    return frame


def draw_results(frame, class_name, confidence, class_names, all_probs,
                 frozen=False, fruit_found=True, fruit_boxes=None):
    """Overlay live prediction results onto the frame."""
    h, w = frame.shape[:2]

    # Coloured border — green = fresh, red = spoiled, orange = frozen, grey = no fruit
    if frozen:
        border_color = (255, 180, 0)
        border_thickness = 6
    elif not fruit_found:
        border_color = (80, 80, 80)
        border_thickness = 2
    elif class_name and "fresh" in class_name.lower():
        border_color = (0, 220, 0)
        border_thickness = 4
    elif class_name:
        border_color = (0, 0, 220)
        border_thickness = 4
    else:
        border_color = (100, 100, 100)
        border_thickness = 2
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, border_thickness)

    # Draw detected fruit bounding boxes — coordinates are full-frame since
    # detection now runs on the full frame, not the crop
    if fruit_boxes and not frozen:
        for (bx1, by1, bx2, by2) in fruit_boxes:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 180), 2)

    # Status badge top-left
    mode_text = "❚❚ FROZEN — SPACE to resume" if frozen else "● LIVE"
    mode_color = (0, 200, 255) if frozen else (0, 255, 100)
    frame = put_text_with_background(
        frame, mode_text, (10, 32), scale=0.65, color=mode_color,
        bg_color=(0, 0, 0), alpha=0.7
    )

    # Guide box
    if not frozen:
        box_size = min(h, w) * 3 // 4
        x1, y1 = (w - box_size) // 2, (h - box_size) // 2
        x2, y2 = x1 + box_size, y1 + box_size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(frame, "Position fruit here",
                    (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Main prediction label
    if not fruit_found and not frozen:
        frame = put_text_with_background(
            frame, "No fruit detected", (10, h - 55),
            scale=0.85, color=(140, 140, 140), bg_color=(0, 0, 0), alpha=0.65
        )
    elif class_name:
        pred_color = (0, 255, 0) if "fresh" in class_name.lower() else (0, 60, 255)
        result_text = f"{class_name.upper()}  {confidence * 100:.1f}%"
        frame = put_text_with_background(
            frame, result_text, (10, h - 55), scale=1.05, color=pred_color,
            thickness=2, bg_color=(0, 0, 0), alpha=0.75
        )

        # Probability bars for each class
        y_off = h - 85
        for cls, prob in zip(class_names, all_probs):
            bar_len = int(prob * 160)
            bar_color = (0, 210, 0) if "fresh" in cls.lower() else (0, 50, 210)
            cv2.rectangle(frame, (10, y_off - 4), (10 + bar_len, y_off + 10), bar_color, -1)
            frame = put_text_with_background(
                frame, f"{cls}: {prob * 100:.1f}%", (178, y_off + 9),
                scale=0.48, color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.5
            )
            y_off -= 24
    else:
        frame = put_text_with_background(
            frame, "Waiting for first result...", (10, h - 55),
            scale=0.7, color=(180, 180, 180), bg_color=(0, 0, 0), alpha=0.6
        )

    # Bottom instruction bar
    frame = put_text_with_background(
        frame, "SPACE: Freeze/Unfreeze  |  Q / ESC: Quit",
        (10, h - 12), scale=0.52, color=(180, 180, 180), bg_color=(0, 0, 0), alpha=0.55
    )

    return frame


# ── Background inference thread ────────────────────────────────────────────────

class LiveClassifier:
    """
    Runs model inference on a background thread so the camera loop
    always stays smooth regardless of classification speed.
    """

    def __init__(self, model, transform, class_names, device,
                 fruit_detector=None, interval=1.0, tta_n=5):
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.device = device
        self.fruit_detector = fruit_detector
        self.interval = interval
        self.tta_n = tta_n

        self._lock = threading.Lock()
        self._latest_frame = None         # full frame — used for fruit detection
        self._latest_crop = None          # centre crop — used for freshness classification
        self._result = None               # (class_name, confidence, probs, fruit_found, boxes)
        self._running = False
        self._thread = None

        # Updated TTA transform to match baseline.py's augmented transforms
        self.tta_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),  # Match baseline's full rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                 saturation=0.3, hue=0.1),  # Match baseline's increased jitter
            transforms.RandomGrayscale(p=0.05),  # Added to match baseline
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def feed(self, full_frame: np.ndarray, crop: np.ndarray):
        """Camera loop hands off the full frame (for detection) and centre crop (for classification)."""
        with self._lock:
            self._latest_frame = full_frame.copy()
            self._latest_crop = crop.copy()

    def get_result(self):
        """Returns (class_name, confidence, probs, fruit_found, boxes) or None if not ready yet."""
        with self._lock:
            return self._result

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            with self._lock:
                full_frame = self._latest_frame
                crop = self._latest_crop

            if full_frame is not None and crop is not None:
                # Step 1: detect fruit on the full frame so nothing is cropped out
                if self.fruit_detector is not None:
                    fruit_found, boxes = self.fruit_detector.contains_fruit(full_frame)
                else:
                    fruit_found, boxes = True, []

                # Step 2: only run freshness classifier if fruit is present
                if fruit_found:
                    class_name, conf, probs = self._classify(crop)
                else:
                    class_name, conf, probs = None, 0.0, []

                with self._lock:
                    self._result = (class_name, conf, probs, fruit_found, boxes)

            time.sleep(self.interval)

    def _classify(self, bgr_crop):
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            # Original (clean) pass
            t = self.transform(pil).unsqueeze(0).to(self.device)
            all_probs.append(torch.softmax(self.model(t), dim=1))
            # TTA passes with augmented transforms
            for _ in range(self.tta_n - 1):
                t = self.tta_transform(pil).unsqueeze(0).to(self.device)
                all_probs.append(torch.softmax(self.model(t), dim=1))

        avg = torch.stack(all_probs).mean(0)
        conf, pred = torch.max(avg, dim=1)
        return self.class_names[pred.item()], conf.item(), avg.cpu().numpy()[0]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live fruit freshness detection")
    parser.add_argument("--model", default="best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--interval", type=float, default=CLASSIFY_INTERVAL,
                        help="Seconds between classifications (default: 1.0)")
    parser.add_argument("--tta", type=int, default=5,
                        help="Number of TTA augmentations (default: 5)")
    args = parser.parse_args()

    # ── Load model ──
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please train first with: python baseline.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} on {device}...")
    try:
        # Updated to handle the new checkpoint format from baseline.py
        model, ckpt = load_model_from_checkpoint(model_path, map_location=device)
        model = model.to(device)
        class_names = ckpt["class_names"]
        arch = ckpt.get("arch", "unknown")
        print(f"Model architecture: {arch}")
        print(f"Classes: {class_names}")
        print(f"Best validation accuracy: {ckpt.get('val_accuracy', 'N/A')}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    transform = get_val_transform(image_size=224)

    # ── Load fruit detector ──
    try:
        fruit_detector = FruitDetector(device)
    except Exception as e:
        print(f"Warning: fruit detector failed to load ({e}). Running without it.")
        fruit_detector = None

    # ── Open camera ──
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    print("\n" + "=" * 50)
    print("SpoiledOrNot — Live Fruit Freshness Detector")
    print("=" * 50)
    print("  Hold a fruit inside the guide box.")
    print("  Results update automatically every second.")
    print("  SPACE = freeze frame | Q / ESC = quit")
    print(f"  TTA (Test Time Augmentation): {args.tta} passes")
    print("=" * 50 + "\n")

    # ── Start background classifier ──
    classifier = LiveClassifier(
        model, transform, class_names, device,
        fruit_detector=fruit_detector, interval=args.interval,
        tta_n=args.tta  # Allow TTA count to be configurable
    )
    classifier.start()

    frozen = False
    frozen_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame — retrying…")
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        box_size = min(h, w) * 3 // 4
        x1, y1 = (w - box_size) // 2, (h - box_size) // 2
        x2, y2 = x1 + box_size, y1 + box_size

        if not frozen:
            # Feed the full frame to the detector and the centre crop to the classifier
            crop = frame[y1:y2, x1:x2]
            classifier.feed(frame, crop)
            display = frame.copy()
        else:
            display = frozen_frame.copy()

        # Fetch latest result (may be None until first inference finishes)
        result = classifier.get_result()
        if result:
            class_name, confidence, all_probs, fruit_found, fruit_boxes = result
        else:
            class_name, confidence, all_probs, fruit_found, fruit_boxes = None, 0.0, [], True, []

        display = draw_results(display, class_name, confidence, class_names, all_probs,
                               frozen, fruit_found, fruit_boxes)
        cv2.imshow("SpoiledOrNot — Live Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            frozen = not frozen
            if frozen:
                frozen_frame = frame.copy()
                print(f"Frozen — last result: {class_name} ({confidence*100:.1f}%)" if class_name else "Frozen.")
            else:
                print("Resumed live feed.")

    classifier.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed. Goodbye!")


if __name__ == "__main__":
    main()