"""
Real-time fruit freshness detection using laptop camera with Gemma 4 multi-modal LLM integration.
Classifies continuously on the live feed - no need to press SPACE.
Press SPACE to freeze/unfreeze, ESC or 'q' to quit.
Press 'g' to get AI-powered analysis from Gemma 4.

Run: python camera_detect.py
     python camera_detect.py --model best_model.pt
"""

import argparse
import base64
import io
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import requests
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
PRODUCE_COCO_IDS = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57}
PRODUCE_DETECT_THRESHOLD = 0.4
FRUIT_DETECT_THRESHOLD = 0.4

# Ollama Gemma 4 configuration
OLLAMA_URL = "http://localhost:11434"
GEMMA_MODEL = "gemma4:26b"  # Use your actual model name


# ------------------------------------------------------------------------------
# GEMMA 4 MULTI-MODAL LLM INTEGRATION
# ------------------------------------------------------------------------------

class GemmaAnalyzer:
    """
    Multi-modal analyzer using Gemma 4 via Ollama.
    Combines CV model predictions with LLM-powered visual analysis.
    Includes detailed status tracking for user feedback.
    """

    # Status constants
    STATUS_IDLE = "idle"
    STATUS_CONNECTING = "connecting"
    STATUS_ENCODING = "encoding_image"
    STATUS_SENDING = "sending_request"
    STATUS_WAITING = "waiting_response"
    STATUS_RECEIVING = "receiving_response"
    STATUS_DONE = "done"
    STATUS_ERROR = "error"
    STATUS_TIMEOUT = "timeout"

    def __init__(self, model_name: str = GEMMA_MODEL, ollama_url: str = OLLAMA_URL):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.available = self._check_availability()
        self._lock = threading.Lock()
        self._analyzing = False
        self._latest_analysis: Optional[str] = None
        self._status = self.STATUS_IDLE
        self._status_message = "Ready"
        self._progress_percent = 0
        self._last_error: Optional[str] = None
        self._start_time: Optional[float] = None

    def _check_availability(self) -> bool:
        """Check if Ollama is running and any Gemma model is available."""
        try:
            print("Checking Ollama availability...")
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [m.get("name", "") for m in models]
                print(f"Available models: {available_models}")
                # Use exact match first, then fall back to any gemma variant
                if any(self.model_name == name for name in available_models):
                    print(f"Gemma model ({self.model_name}) is available!")
                    return True
                gemma_variants = [n for n in available_models if "gemma" in n.lower()]
                if gemma_variants:
                    preferred = [n for n in gemma_variants if "gemma4" in n.lower() or "gemma:4" in n.lower()]
                    self.model_name = preferred[0] if preferred else gemma_variants[0]
                    print(f"Using Gemma model: '{self.model_name}'")
                    return True
                print(f"No Gemma model found. Available: {available_models}")
                print(f"Install with: ollama pull {self.model_name}")
                return False
            return False
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to Ollama at {self.ollama_url}")
            print("Make sure Ollama is running: ollama serve")
            return False
        except Exception as e:
            print(f"Gemma 4 not available: {e}")
            return False

    def is_available(self) -> bool:
        return self.available

    def get_status(self) -> Tuple[str, str, int]:
        """Get current status: (status_code, message, progress_percent)."""
        with self._lock:
            return self._status, self._status_message, self._progress_percent

    def get_last_error(self) -> Optional[str]:
        """Get the last error message if any."""
        with self._lock:
            return self._last_error

    def is_analyzing(self) -> bool:
        """Check if analysis is currently in progress."""
        with self._lock:
            return self._analyzing

    def _update_status(self, status: str, message: str, progress: int = 0):
        """Update status thread-safely."""
        with self._lock:
            self._status = status
            self._status_message = message
            self._progress_percent = progress
        # Also print to console for CLI users
        elapsed = ""
        if self._start_time:
            elapsed = f" ({time.time() - self._start_time:.1f}s)"
        print(f"[Gemma]{elapsed} {message}")

    def analyze_frame(
        self,
        frame: np.ndarray,
        class_name: Optional[str],
        confidence: float,
        produce_type: str
    ) -> Optional[str]:
        """
        Analyze a frame using Gemma 4 multi-modal capabilities.
        Returns natural language analysis with recommendations.
        """
        if not self.available:
            self._update_status(self.STATUS_ERROR, "Gemma 4 not available. Run: ollama pull gemma:4b")
            return None

        if self._analyzing:
            self._update_status(self.STATUS_WAITING, "Analysis already in progress...")
            return self._latest_analysis

        with self._lock:
            self._analyzing = True
            self._start_time = time.time()
            self._last_error = None

        try:
            # Step 1: Encode image
            self._update_status(self.STATUS_ENCODING, "Converting image for Gemma 4...", 10)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Step 2: Build prompt
            self._update_status(self.STATUS_ENCODING, "Preparing AI prompt...", 20)
            freshness_status = class_name if class_name else "Unknown"
            confidence_pct = confidence * 100
            prompt = self._build_analysis_prompt(freshness_status, confidence_pct, produce_type)

            # Step 3: Send request
            self._update_status(self.STATUS_SENDING, f"Sending to {self.model_name} (this may take 10-30s)...", 30)

            # Step 4: Wait for response with progress updates
            self._update_status(self.STATUS_WAITING, "Gemma 4 is analyzing the image...", 50)

            # Use /api/chat — more reliable for multimodal than /api/generate
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [img_base64],
                        }
                    ],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2048
                    }
                },
                timeout=120  # Large models need more time on first run
            )

            # Step 5: Process response
            self._update_status(self.STATUS_RECEIVING, "Processing AI response...", 80)

            if response.status_code == 200:
                result = response.json()
                # /api/chat returns message.content; fall back to response field
                message = result.get("message", {})
                analysis = (message.get("content") or result.get("response") or "").strip()

                if analysis:
                    elapsed = time.time() - self._start_time if self._start_time else 0
                    self._update_status(self.STATUS_DONE, f"Analysis complete! ({elapsed:.1f}s)", 100)
                    with self._lock:
                        self._latest_analysis = analysis
                    return analysis
                else:
                    print(f"[Gemma] Raw response JSON: {result}")
                    self._update_status(self.STATUS_ERROR, "Empty response from Gemma 4 — check Ollama logs", 0)
                    self._last_error = "Empty response from model"

            elif response.status_code == 404:
                error_msg = f"Model '{self.model_name}' not found. Install with: ollama pull {self.model_name}"
                self._update_status(self.STATUS_ERROR, error_msg, 0)
                self._last_error = error_msg

            else:
                print(f"[Gemma] Error body: {response.text[:500]}")
                error_msg = f"Ollama error {response.status_code}: {response.text[:100]}"
                self._update_status(self.STATUS_ERROR, error_msg, 0)
                self._last_error = error_msg

        except requests.exceptions.Timeout:
            elapsed = time.time() - self._start_time if self._start_time else 0
            error_msg = f"Request timed out after {elapsed:.1f}s. Gemma may be loading or busy."
            self._update_status(self.STATUS_TIMEOUT, error_msg, 0)
            self._last_error = error_msg
            print(f"\nTroubleshooting:")
            print("  - First request loads the model into memory (can take 60-120s)")
            print("  - Try again - subsequent requests will be faster")
            print("  - Check if Ollama is still running: curl http://localhost:11434/api/tags")

        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Ollama. Is it running?"
            self._update_status(self.STATUS_ERROR, error_msg, 0)
            self._last_error = error_msg
            print(f"\nTo start Ollama: ollama serve")

        except Exception as e:
            elapsed = time.time() - self._start_time if self._start_time else 0
            error_msg = f"Error after {elapsed:.1f}s: {str(e)}"
            self._update_status(self.STATUS_ERROR, error_msg, 0)
            self._last_error = error_msg

        finally:
            with self._lock:
                self._analyzing = False
                self._start_time = None

        return self._latest_analysis

    def _build_analysis_prompt(
        self,
        freshness_status: str,
        confidence: float,
        produce_type: str
    ) -> str:
        """Build a context-aware prompt for Gemma 4."""

        base_prompt = f"""You are a produce freshness expert. Analyze this image of {produce_type.lower() if produce_type != 'Unknown' else 'produce'}.

CV Model Detection Results:
- Freshness Status: {freshness_status}
- Confidence: {confidence:.1f}%
- Produce Type: {produce_type}

Provide a brief analysis (2-3 sentences) covering:
1. Visual signs of freshness or spoilage you observe
2. Specific recommendations for storage or use
3. Safety assessment (safe to eat, use soon, or discard)

Keep your response concise and practical."""

        return base_prompt

    def get_cached_analysis(self) -> Optional[str]:
        """Get the most recent analysis without making a new request."""
        with self._lock:
            return self._latest_analysis

    def clear_cache(self):
        """Clear the cached analysis."""
        with self._lock:
            self._latest_analysis = None


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


def wrap_text(text: str, max_chars: int = 50) -> List[str]:
    """Wrap text into lines of maximum length."""
    words = text.split()
    lines = []
    current_line = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 <= max_chars:
            current_line.append(word)
            current_len += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def draw_results(
    frame: np.ndarray,
    class_name: Optional[str],
    confidence: float,
    class_names: list,
    all_probs: List[float],
    frozen: bool = False,
    fruit_found: bool = True,
    fruit_boxes: Optional[List[List[int]]] = None,
    produce_type: str = "Unknown",
    gemma_analysis: Optional[str] = None,
    show_analysis: bool = False,
    gemma_analyzer: Optional[GemmaAnalyzer] = None
) -> np.ndarray:
    """Overlay live prediction results onto the frame."""
    h, w = frame.shape[:2]

    # Check if fresh
    is_fresh = class_name and ("fresh" in class_name.lower() or class_name.lower() == "fresh")

    # Coloured border
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

    # Gemma analysis indicator with status
    if gemma_analyzer and gemma_analyzer.is_analyzing():
        status, status_msg, progress = gemma_analyzer.get_status()
        progress_bar = "█" * (progress // 10) + "░" * (10 - progress // 10)
        status_text = f"🤖 {status_msg} [{progress_bar}] {progress}%"
        frame = put_text_with_background(
            frame, status_text, (10, 58), scale=0.5,
            color=(255, 200, 100), bg_color=(0, 0, 0), alpha=0.8
        )
    elif gemma_analysis:
        frame = put_text_with_background(
            frame, "🤖 AI Analysis Ready (press 'g')", (10, 58), scale=0.55,
            color=(255, 200, 0), bg_color=(0, 0, 0), alpha=0.7
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

    # Display Gemma analysis panel if enabled
    if show_analysis:
        # Draw semi-transparent panel on the right side
        panel_width = min(400, w // 2)
        panel_x = w - panel_width - 10
        panel_y = 80
        panel_height = min(420, h - panel_y - 10)

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                      (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        # Panel border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                      (100, 100, 150), 1)

        # Title
        cv2.putText(frame, "🤖 Gemma 4 Analysis", (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)

        # Show content based on state
        if gemma_analyzer and gemma_analyzer.is_analyzing():
            # Show progress
            status, status_msg, progress = gemma_analyzer.get_status()
            cv2.putText(frame, "Analyzing...", (panel_x + 10, panel_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 1)
            progress_bar = "█" * (progress // 10) + "░" * (10 - progress // 10)
            cv2.putText(frame, f"{progress}% {progress_bar}", (panel_x + 10, panel_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            # Show truncated status message
            msg = status_msg[:45] + "..." if len(status_msg) > 45 else status_msg
            cv2.putText(frame, msg, (panel_x + 10, panel_y + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        elif gemma_analyzer and gemma_analyzer.get_last_error():
            # Show error
            error_msg = gemma_analyzer.get_last_error()
            cv2.putText(frame, "Analysis Failed:", (panel_x + 10, panel_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 1)
            wrapped_lines = wrap_text(error_msg, max_chars=45)
            y_offset = panel_y + 80
            for line in wrapped_lines[:5]:
                cv2.putText(frame, line, (panel_x + 10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 100, 100), 1)
                y_offset += 22
        elif gemma_analysis:
            # Show analysis result — fit as many lines as the panel allows.
            # max_chars is derived from usable panel width: (panel_width - 20px padding) / ~7.5px per char at scale 0.45
            font_scale = 0.45
            char_width_px = 7.5
            usable_width = panel_width - 24
            max_chars = max(20, int(usable_width / char_width_px))
            line_height = 18
            content_start = panel_y + 55
            max_lines = (panel_height - 55 - 10) // line_height
            wrapped_lines = wrap_text(gemma_analysis, max_chars=max_chars)
            y_offset = content_start
            for line in wrapped_lines[:max_lines]:
                cv2.putText(frame, line, (panel_x + 10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1)
                y_offset += line_height
            if len(wrapped_lines) > max_lines:
                cv2.putText(frame, "...", (panel_x + 10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (160, 160, 160), 1)
        else:
            # Waiting state
            cv2.putText(frame, "Starting analysis...", (panel_x + 10, panel_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Bottom instruction bar
    instructions = "SPACE: Freeze/Unfreeze | G: AI Analysis | Q/ESC: Quit"
    frame = put_text_with_background(
        frame, instructions,
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
    Includes Gemma 4 multi-modal analysis.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transform: transforms.Compose,
        class_names: list,
        device: torch.device,
        fruit_detector: Optional[FruitDetector] = None,
        gemma_analyzer: Optional[GemmaAnalyzer] = None,
        interval: float = CLASSIFY_INTERVAL,
        tta_n: int = 1
    ):
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.device = device
        self.fruit_detector = fruit_detector
        self.gemma_analyzer = gemma_analyzer
        self.interval = interval
        self.tta_n = tta_n

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_crop: Optional[np.ndarray] = None
        self._result: Optional[Tuple] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._gemma_analysis: Optional[str] = None

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
            self._latest_frame = full_frame
            self._latest_crop = crop.copy()

    def get_result(self) -> Optional[Tuple]:
        """Returns (class_name, confidence, probs, fruit_found, boxes, produce_type, gemma_analysis) or None."""
        with self._lock:
            if self._result:
                return (*self._result, self._gemma_analysis)
            return None

    def request_gemma_analysis(self) -> bool:
        """Request a new Gemma 4 analysis of the current frame."""
        if not self.gemma_analyzer or not self.gemma_analyzer.is_available():
            return False

        with self._lock:
            frame = self._latest_frame
            result = self._result

        if frame is None or result is None:
            return False

        class_name, confidence, _, fruit_found, _, produce_type = result

        # Run analysis in background thread to not block
        def analyze():
            analysis = self.gemma_analyzer.analyze_frame(
                frame, class_name, confidence, produce_type
            )
            if analysis:
                with self._lock:
                    self._gemma_analysis = analysis

        threading.Thread(target=analyze, daemon=True).start()
        return True

    def clear_gemma_analysis(self):
        """Clear the Gemma analysis cache."""
        with self._lock:
            self._gemma_analysis = None
        if self.gemma_analyzer:
            self.gemma_analyzer.clear_cache()

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
            if now - last_classify < self.interval:
                time.sleep(0.05)
                continue

            with self._lock:
                full_frame = self._latest_frame
                crop = self._latest_crop

            if full_frame is not None and crop is not None:
                # Step 1: detect fruit on the full frame
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

                # Note: Auto-trigger disabled to prevent spam on errors
                # User must manually press 'g' to request analysis
                pass

            last_classify = time.time()

    def _classify(self, bgr_crop: np.ndarray) -> Tuple[str, float, np.ndarray, str]:
        """Classify a cropped image and return (freshness_status, confidence, probabilities, produce_type)."""
        small = cv2.resize(bgr_crop, (320, 320))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            t = self.transform(pil).unsqueeze(0).to(self.device)
            all_probs.append(torch.softmax(self.model(t), dim=1))
            for _ in range(self.tta_n - 1):
                t = self.tta_transform(pil).unsqueeze(0).to(self.device)
                all_probs.append(torch.softmax(self.model(t), dim=1))

        avg = torch.stack(all_probs).mean(0)
        conf, pred = torch.max(avg, dim=1)
        raw_class = self.class_names[pred.item()]

        normalized_class, produce_type = self._normalize_freshness_class(raw_class, avg.cpu().numpy()[0])
        return normalized_class, conf.item(), avg.cpu().numpy()[0], produce_type

    def _normalize_freshness_class(self, raw_class: str, probs: np.ndarray) -> Tuple[str, str]:
        """Normalize class name to Fresh/Rotten and extract produce type."""
        raw_lower = raw_class.lower()
        produce_type = "Unknown"

        produce_mapping = {
            "freshapples": "Apple",
            "freshapple": "Apple",
            "freshbanana": "Banana",
            "freshcucumber": "Cucumber",
            "freshokra": "Okra",
            "freshoranges": "Orange",
            "freshorange": "Orange",
            "freshpotato": "Potato",
            "freshtomato": "Tomato",
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

        if raw_lower in produce_mapping:
            produce_type = produce_mapping[raw_lower]
        else:
            known_produce = ["apple", "banana", "cucumber", "okra", "orange", "potato", "tomato"]
            for produce in known_produce:
                if produce in raw_lower:
                    produce_type = produce.capitalize()
                    break
            if produce_type == "Unknown":
                cleaned = raw_lower.replace("fresh", "").replace("rotten", "").replace("stale", "")
                if cleaned:
                    if cleaned.endswith('s'):
                        cleaned = cleaned[:-1]
                    produce_type = cleaned.capitalize()

        has_fresh = "fresh" in raw_lower
        has_rotten = any(word in raw_lower for word in ["rotten", "stale", "spoiled", "bad"])

        if has_fresh and not has_rotten:
            return "Fresh", produce_type
        elif has_rotten and not has_fresh:
            return "Rotten", produce_type
        elif has_fresh and has_rotten:
            return raw_class.replace("_", " ").title(), produce_type

        if len(self.class_names) == 2:
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

        return raw_class.replace("_", " ").title(), produce_type


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main() -> None:
    """Main entry point for live fruit freshness detection with Gemma 4."""
    parser = argparse.ArgumentParser(description="Live fruit freshness detection with Gemma 4 AI")
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
    parser.add_argument(
        "--no-gemma",
        action="store_true",
        help="Disable Gemma 4 multi-modal analysis"
    )
    args = parser.parse_args()

    frame_interval = 1.0 / args.fps

    # Load CV model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please train first with: python baseline.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CV model from {model_path} on {device}...")
    try:
        model, ckpt = load_model_from_checkpoint(model_path, map_location=device)
        model = model.to(device)
        class_names = ckpt["class_names"]
        arch = ckpt.get("arch", "unknown")
        print(f"CV Model architecture: {arch}")
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

    # Initialize Gemma 4 analyzer
    gemma_analyzer = None
    if not args.no_gemma:
        print("\nInitializing Gemma 4 multi-modal analyzer...")
        gemma_analyzer = GemmaAnalyzer()
        if gemma_analyzer.is_available():
            print("Gemma 4 is ready for AI-powered analysis!")
            print("Press 'G' during detection to get detailed insights.")
        else:
            print("Gemma 4 not available. Make sure Ollama is running with Gemma 4 installed.")
            print("To install: ollama pull gemma:4b")
            gemma_analyzer = None

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

    print("\n" + "=" * 60)
    print("SpoiledOrNot - Live Fruit Freshness Detector")
    print("           with Gemma 4 Multi-Modal AI")
    print("=" * 60)
    print("  Hold a fruit inside the guide box.")
    print(f"  Results update every {args.interval}s | Display: {args.fps} FPS")
    print("  SPACE = freeze frame | G = AI analysis | Q / ESC = quit")
    print("=" * 60 + "\n")

    # Start background classifier
    classifier = LiveClassifier(
        model, transform, class_names, device,
        fruit_detector=fruit_detector,
        gemma_analyzer=gemma_analyzer,
        interval=args.interval,
        tta_n=args.tta,
    )
    classifier.start()

    frozen = False
    frozen_frame: Optional[np.ndarray] = None
    show_analysis = False
    last_frame_time = 0.0

    while True:
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

        # Rotate frame if needed
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
            class_name, confidence, all_probs, fruit_found, fruit_boxes, produce_type, gemma_analysis = result
        else:
            class_name, confidence, all_probs, fruit_found, fruit_boxes, produce_type, gemma_analysis = None, 0.0, [], True, [], "Unknown", None

        display = draw_results(display, class_name, confidence, class_names, all_probs,
                               frozen, fruit_found, fruit_boxes, produce_type,
                               gemma_analysis, show_analysis, gemma_analyzer)
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
                show_analysis = False
        elif key == ord("g") or key == ord("G"):
            if gemma_analyzer and gemma_analyzer.is_available():
                if frozen and frozen_frame is not None:
                    if gemma_analyzer.is_analyzing():
                        # Show current status
                        status, msg, progress = gemma_analyzer.get_status()
                        print(f"\n🤖 Analysis in progress... ({progress}%)")
                        print(f"   Status: {msg}")
                        show_analysis = True
                    else:
                        print("\n🤖 Requesting Gemma 4 AI analysis...")
                        print("   This typically takes 10-30 seconds on first run")
                        print("   (Gemma 4 needs to load into memory)")
                        show_analysis = not show_analysis
                        if show_analysis:
                            success = classifier.request_gemma_analysis()
                            if not success:
                                print("   ⚠️ Failed to start analysis - check if produce is detected")
                else:
                    print("\n⚠️  Freeze the frame first (SPACE) to analyze!")
            else:
                if gemma_analyzer:
                    error = gemma_analyzer.get_last_error()
                    if error:
                        print(f"\n⚠️  Gemma 4 error: {error}")
                    else:
                        print("\n⚠️  Gemma 4 not available. Install with: ollama pull gemma:4b")
                else:
                    print("\n⚠️  Gemma 4 not available. Install with: ollama pull gemma:4b")

    classifier.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed. Goodbye!")


if __name__ == "__main__":
    main()
