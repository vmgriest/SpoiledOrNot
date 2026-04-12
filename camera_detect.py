"""
Real-time fruit freshness detection using laptop camera.
Classifies continuously on the live feed - no need to press SPACE.
Press SPACE to freeze/unfreeze, ESC or 'q' to quit.

Run: python camera_detect.py
     python camera_detect.py --model best_model.pt
"""

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from baseline import get_val_transform, load_model_from_checkpoint


# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Seconds between each classification (lower = more responsive, higher CPU use)
CLASSIFY_INTERVAL = 1.0  # classify every second for faster results

# Cap the display loop at 30 FPS - smoother video
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS

# COCO class IDs for produce detection (only general fruit/vegetable classes)
# We don't use COCO for specific produce type identification - we use the freshness model's
# class names to determine if it's an Apple, Tomato, Banana, etc.
# This avoids confusion like tomatoes being detected as apples.
PRODUCE_COCO_IDS = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57}
# 52: banana, 53: apple, 55: orange, 57: carrot
# 46-51, 54, 56: other food items that might be produce-like
PRODUCE_DETECT_THRESHOLD = 0.4
FRUIT_DETECT_THRESHOLD = 0.4  # Kept for backward compatibility


# ------------------------------------------------------------------------------
# FRUIT DETECTOR (FASTER R-CNN, COCO)
# ------------------------------------------------------------------------------

class FruitDetector:
    """
    Lightweight wrapper around torchvision's Faster R-CNN (MobileNet backbone).
    Flags detections whose COCO class is in FRUIT_COCO_IDS or PRODUCE_COCO_IDS.
    No internet download needed after the first run - weights are cached locally.
    """

    def __init__(self, device: torch.device, threshold: float = FRUIT_DETECT_THRESHOLD):
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )
        print("Loading produce detector (Faster R-CNN MobileNet)...")
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self._model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        self._model.eval().to(device)
        self._transform = weights.transforms()
        self._device = device
        self._threshold = threshold
        print("Produce detector ready.")

    @torch.no_grad()
    def contains_fruit(self, bgr_frame: np.ndarray) -> Tuple[bool, list]:
        """
        Returns (fruit_found: bool, boxes: list of (x1,y1,x2,y2)).
        Runs on the full frame so nothing is accidentally cropped out.
        Only detects if produce is present - the freshness model determines the type.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self._device)
        preds = self._model(tensor)[0]

        boxes = []
        for label, score, box in zip(preds["labels"], preds["scores"], preds["boxes"]):
            label_id = label.item()
            # Check if this is a food/produce item with sufficient confidence
            if label_id in PRODUCE_COCO_IDS and score >= PRODUCE_DETECT_THRESHOLD:
                boxes.append(box.cpu().numpy().astype(int).tolist())

        return len(boxes) > 0, boxes


# ------------------------------------------------------------------------------
# TEXT / DRAWING HELPERS
# ------------------------------------------------------------------------------

def put_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.8,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.6
) -> np.ndarray:
    """Draw text with a semi-transparent background."""
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - text_h - 10), (x + text_w + 10, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x + 5, y), font, scale, color, thickness)
    return frame


def draw_results(
    frame: np.ndarray,
    class_name: Optional[str],
    confidence: float,
    class_names: list,
    all_probs: List[float],
    frozen: bool = False,
    fruit_found: bool = True,
    fruit_boxes: Optional[List[List[int]]] = None,
    produce_type: str = "Unknown"
) -> np.ndarray:
    """Overlay live prediction results onto the frame."""
    h, w = frame.shape[:2]

    # Check if fresh (handles normalized "Fresh" or original "Apple_Fresh" etc.)
    is_fresh = class_name and ("fresh" in class_name.lower() or class_name.lower() == "fresh")

    # Coloured border - green = fresh, red = spoiled, orange = frozen, grey = no fruit
    if frozen:
        border_color = (255, 180, 0)
        border_thickness = 6
    elif not fruit_found:
        border_color = (80, 80, 80)
        border_thickness = 2
    elif is_fresh:
        border_color = (0, 220, 0)
        border_thickness = 4
    elif class_name:
        border_color = (0, 0, 220)
        border_thickness = 4
    else:
        border_color = (100, 100, 100)
        border_thickness = 2
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, border_thickness)

    # Draw detected produce bounding boxes
    if fruit_boxes and not frozen:
        for (bx1, by1, bx2, by2) in fruit_boxes:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 180), 2)

    # Status badge top-left
    mode_text = "|| FROZEN - SPACE to resume" if frozen else "LIVE"
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
        cv2.putText(frame, "Position produce here",
                    (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Main prediction label
    if not fruit_found and not frozen:
        frame = put_text_with_background(
            frame, "No produce detected", (10, h - 55),
            scale=0.85, color=(140, 140, 140), bg_color=(0, 0, 0), alpha=0.65
        )
    elif class_name:
        pred_color = (0, 255, 0) if is_fresh else (0, 60, 255)
        result_text = f"{class_name.upper()}  {confidence * 100:.1f}%"
        frame = put_text_with_background(
            frame, result_text, (10, h - 55), scale=1.05, color=pred_color,
            thickness=2, bg_color=(0, 0, 0), alpha=0.75
        )
        # Show produce type below freshness result
        if produce_type and produce_type != "Unknown":
            type_text = f"Type: {produce_type}"
            frame = put_text_with_background(
                frame, type_text, (10, h - 88), scale=0.75, color=(0, 200, 255),
                thickness=2, bg_color=(0, 0, 0), alpha=0.7
            )
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


# ------------------------------------------------------------------------------
# BACKGROUND INFERENCE THREAD
# ------------------------------------------------------------------------------

class LiveClassifier:
    """
    Runs model inference on a background thread so the camera loop
    always stays smooth regardless of classification speed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transform: transforms.Compose,
        class_names: list,
        device: torch.device,
        fruit_detector: Optional[FruitDetector] = None,
        interval: float = CLASSIFY_INTERVAL,
        tta_n: int = 1
    ):
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.device = device
        self.fruit_detector = fruit_detector
        self.interval = interval
        self.tta_n = tta_n

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None         # full frame - used for fruit detection
        self._latest_crop: Optional[np.ndarray] = None          # centre crop - used for freshness classification
        self._result: Optional[Tuple] = None                    # (class_name, confidence, probs, fruit_found, boxes, produce_type)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.tta_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def feed(self, full_frame: np.ndarray, crop: np.ndarray) -> None:
        """Camera loop hands off the full frame (for detection) and centre crop (for classification)."""
        with self._lock:
            self._latest_frame = full_frame  # No copy - we don't modify it
            self._latest_crop = crop.copy()  # Only crop needs copy as it might be used elsewhere

    def get_result(self) -> Optional[Tuple]:
        """Returns (class_name, confidence, probs, fruit_found, boxes) or None if not ready yet."""
        with self._lock:
            return self._result

    def start(self) -> None:
        """Start the background classification thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background classification thread."""
        self._running = False

    def _loop(self) -> None:
        """Main classification loop running in background thread."""
        last_classify = 0
        while self._running:
            now = time.time()
            # Sleep cheaply until the interval has elapsed - no spinning
            if now - last_classify < self.interval:
                time.sleep(0.05)
                continue

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
                    class_name, conf, probs, produce_type = self._classify(crop)
                else:
                    class_name, conf, probs, produce_type = None, 0.0, [], "Unknown"

                with self._lock:
                    self._result = (class_name, conf, probs, fruit_found, boxes, produce_type)

            last_classify = time.time()

    def _classify(self, bgr_crop: np.ndarray) -> Tuple[str, float, np.ndarray, str]:
        """Classify a cropped image and return (freshness_status, confidence, probabilities, produce_type)."""
        # Downscale crop before conversion - the transform resizes to 224 anyway,
        # so working from 320px instead of full resolution saves time with no accuracy loss
        small = cv2.resize(bgr_crop, (320, 320))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            # Original (clean) pass
            t = self.transform(pil).unsqueeze(0).to(self.device)
            all_probs.append(torch.softmax(self.model(t), dim=1))
            # TTA passes (default tta_n=1 skips these entirely)
            for _ in range(self.tta_n - 1):
                t = self.tta_transform(pil).unsqueeze(0).to(self.device)
                all_probs.append(torch.softmax(self.model(t), dim=1))

        avg = torch.stack(all_probs).mean(0)
        conf, pred = torch.max(avg, dim=1)
        raw_class = self.class_names[pred.item()]

        # Normalize class name to Fresh/Rotten for display and extract produce type
        # This handles datasets with format like "Apple_Fresh", "Banana_Rotten", etc.
        normalized_class, produce_type = self._normalize_freshness_class(raw_class, avg.cpu().numpy()[0])
        return normalized_class, conf.item(), avg.cpu().numpy()[0], produce_type

    def _normalize_freshness_class(self, raw_class: str, probs: np.ndarray) -> Tuple[str, str]:
        """
        Normalize class name to Fresh/Rotten regardless of produce type.
        Also extracts the produce type (Apple, Tomato, Banana, Potato, etc.) from the class name.
        Handles formats like: 'freshpotato', 'freshtomato', 'rottenapples', etc.

        Returns: (freshness_status, produce_type)
        """
        raw_lower = raw_class.lower()
        produce_type = "Unknown"

        # Direct mapping from your actual class names to display names
        produce_mapping = {
            # Fresh produce
            "freshapples": "Apple",
            "freshapple": "Apple",
            "freshbanana": "Banana",
            "freshcucumber": "Cucumber",
            "freshokra": "Okra",
            "freshoranges": "Orange",
            "freshorange": "Orange",
            "freshpotato": "Potato",
            "freshtomato": "Tomato",
            
            # Rotten produce
            "rottenapples": "Apple",
            "rottenapple": "Apple",
            "rottenbanana": "Banana",
            "rottencucumber": "Cucumber",
            "rottenokra": "Okra",
            "rottenoranges": "Orange",
            "rottenorange": "Orange",
            "rottenpotato": "Potato",
            "rottentomato": "Tomato",
        }

        # Check if the raw class exactly matches any of our mappings
        if raw_lower in produce_mapping:
            produce_type = produce_mapping[raw_lower]
        else:
            # Try to extract produce type by checking for known patterns
            known_produce = ["apple", "banana", "cucumber", "okra", "orange", "potato", "tomato"]
            
            for produce in known_produce:
                if produce in raw_lower:
                    produce_type = produce.capitalize()
                    break
            
            # If still unknown, try to extract by removing freshness prefix/suffix
            if produce_type == "Unknown":
                # Remove common freshness words
                cleaned = raw_lower.replace("fresh", "").replace("rotten", "").replace("stale", "")
                if cleaned:
                    # Remove trailing 's' if present (apples -> apple)
                    if cleaned.endswith('s'):
                        cleaned = cleaned[:-1]
                    produce_type = cleaned.capitalize()

        # Determine freshness status
        has_fresh = "fresh" in raw_lower
        has_rotten = any(word in raw_lower for word in ["rotten", "stale", "spoiled", "bad"])

        if has_fresh and not has_rotten:
            return "Fresh", produce_type
        elif has_rotten and not has_fresh:
            return "Rotten", produce_type
        elif has_fresh and has_rotten:
            # Ambiguous - return cleaned raw class
            return raw_class.replace("_", " ").title(), produce_type

        # For binary classification (fallback)
        if len(self.class_names) == 2:
            # Binary classification - check which index has "fresh" vs "rotten"
            fresh_idx = -1
            rotten_idx = -1
            for i, name in enumerate(self.class_names):
                name_lower = name.lower()
                if "fresh" in name_lower:
                    fresh_idx = i
                elif any(word in name_lower for word in ["rotten", "stale", "spoiled"]):
                    rotten_idx = i

            if fresh_idx >= 0 and rotten_idx >= 0:
                freshness = "Fresh" if probs[fresh_idx] > probs[rotten_idx] else "Rotten"
                return freshness, produce_type

        # Default: return cleaned up raw class
        return raw_class.replace("_", " ").title(), produce_type

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main() -> None:
    """Main entry point for live fruit freshness detection."""
    parser = argparse.ArgumentParser(description="Live fruit freshness detection")
    parser.add_argument("--model", default="best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--ip-camera",
        type=str,
        default=None,
        help="IP camera URL (e.g., http://10.0.0.2:4747/video)"
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotate video feed (useful for phone cameras): 0, 90, 180, or 270 degrees"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--interval",
        type=float,
        default=CLASSIFY_INTERVAL,
        help=f"Seconds between classifications (default: {CLASSIFY_INTERVAL})"
    )
    parser.add_argument(
        "--tta",
        type=int,
        default=1,
        help="Number of TTA augmentation passes (default: 1 = disabled for performance)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=TARGET_FPS,
        help=f"Target display FPS (default: {TARGET_FPS})"
    )
    args = parser.parse_args()

    frame_interval = 1.0 / args.fps

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please train first with: python baseline.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} on {device}...")
    try:
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

    # Load fruit detector
    try:
        fruit_detector = FruitDetector(device)
    except Exception as e:
        print(f"Warning: fruit detector failed to load ({e}). Running without it.")
        fruit_detector = None

    # Open camera
    if args.ip_camera:
        print(f"Connecting to IP camera: {args.ip_camera}")
        cap = cv2.VideoCapture(args.ip_camera)
        if not cap.isOpened():
            print(f"Failed to connect to IP camera: {args.ip_camera}")
            print("\nTroubleshooting:")
            print("  - Make sure your phone and computer are on the same WiFi network")
            print("  - Check the IP address in the DroidCam app on your phone")
            print("  - Try opening the URL in a browser first to verify it works")
            sys.exit(1)
        print("IP camera connected successfully!")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Failed to open camera {args.camera}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    print("\n" + "=" * 50)
    print("SpoiledOrNot - Live Fruit Freshness Detector")
    print("=" * 50)
    print("  Hold a fruit inside the guide box.")
    print(f"  Results update every {args.interval}s | Display: {args.fps} FPS | TTA passes: {args.tta}")
    print("  SPACE = freeze frame | Q / ESC = quit")
    print("=" * 50 + "\n")

    # Start background classifier
    classifier = LiveClassifier(
        model, transform, class_names, device,
        fruit_detector=fruit_detector,
        interval=args.interval,
        tta_n=args.tta,
    )
    classifier.start()

    frozen = False
    frozen_frame: Optional[np.ndarray] = None
    last_frame_time = 0.0

    while True:
        # Throttle display loop to TARGET_FPS
        now = time.time()
        elapsed = now - last_frame_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
            continue
        last_frame_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame - retrying...")
            time.sleep(0.1)
            continue

        # Rotate frame if needed (for phone cameras)
        if args.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (args.width, args.height))
        elif args.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif args.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = cv2.resize(frame, (args.width, args.height))

        h, w = frame.shape[:2]
        box_size = min(h, w) * 3 // 4
        x1, y1 = (w - box_size) // 2, (h - box_size) // 2
        x2, y2 = x1 + box_size, y1 + box_size

        if not frozen:
            crop = frame[y1:y2, x1:x2]
            classifier.feed(frame, crop)
            display = frame.copy()
        else:
            display = frozen_frame.copy() if frozen_frame is not None else frame.copy()

        result = classifier.get_result()
        if result:
            class_name, confidence, all_probs, fruit_found, fruit_boxes, produce_type = result
        else:
            class_name, confidence, all_probs, fruit_found, fruit_boxes, produce_type = None, 0.0, [], True, [], "Unknown"

        display = draw_results(display, class_name, confidence, class_names, all_probs,
                               frozen, fruit_found, fruit_boxes, produce_type)
        cv2.imshow("SpoiledOrNot - Live Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            frozen = not frozen
            if frozen:
                frozen_frame = frame.copy()
                if class_name:
                    print(f"Frozen - last result: {class_name} ({confidence*100:.1f}%)")
                else:
                    print("Frozen.")
            else:
                print("Resumed live feed.")

    classifier.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed. Goodbye!")


if __name__ == "__main__":
    main()
