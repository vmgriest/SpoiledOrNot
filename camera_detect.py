"""
Real-time fruit freshness detection using laptop camera.
Press SPACE to capture and classify, ESC or 'q' to quit.

Run: python camera_detect.py
     python camera_detect.py --model best_model.pt
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from baseline import load_model_from_checkpoint, get_val_transform


def put_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                              scale=0.8, color=(255, 255, 255), thickness=2,
                              bg_color=(0, 0, 0), alpha=0.6):
    """Draw text with semi-transparent background for better readability."""
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    # Draw background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - text_h - 10), (x + text_w + 10, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Draw text
    cv2.putText(frame, text, (x + 5, y), font, scale, color, thickness)
    return frame


def classify_frame(frame, model, transform, class_names, device):
    """Convert frame to tensor and run inference."""
    # Convert BGR (OpenCV) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL-like format (H, W, C) -> tensor
    from PIL import Image
    pil_image = Image.fromarray(rgb_frame)

    # Apply transforms
    tensor = transform(pil_image).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    pred_idx = pred.item()
    confidence_val = confidence.item()
    class_name = class_names[pred_idx]

    return class_name, confidence_val, probs.cpu().numpy()[0]


def draw_prediction(frame, class_name, confidence, class_names, all_probs, capture_mode=False):
    """Draw prediction results on the frame."""
    h, w = frame.shape[:2]

    # Status box at top
    status = "CAPTURE MODE" if capture_mode else "PREVIEW MODE"
    frame = put_text_with_background(
        frame, status, (10, 35), scale=0.7, color=(255, 255, 0) if capture_mode else (0, 255, 255),
        bg_color=(0, 0, 0), alpha=0.7
    )

    if class_name:
        # Main prediction
        color = (0, 255, 0) if "fresh" in class_name.lower() else (0, 0, 255)
        result_text = f"{class_name.upper()}: {confidence*100:.1f}%"
        frame = put_text_with_background(
            frame, result_text, (10, h - 60), scale=1.0, color=color,
            thickness=2, bg_color=(0, 0, 0), alpha=0.7
        )

        # All class probabilities
        y_offset = h - 90
        for i, (cls, prob) in enumerate(zip(class_names, all_probs)):
            bar_len = int(prob * 150)
            bar_color = (0, 255, 0) if "fresh" in cls.lower() else (0, 0, 255)
            cv2.rectangle(frame, (10, y_offset - 5), (10 + bar_len, y_offset + 10), bar_color, -1)
            prob_text = f"{cls}: {prob*100:.1f}%"
            frame = put_text_with_background(
                frame, prob_text, (170, y_offset + 8), scale=0.5,
                color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.5
            )
            y_offset -= 25

    # Instructions
    inst_text = "SPACE: Capture | ESC or Q: Quit"
    frame = put_text_with_background(
        frame, inst_text, (10, h - 20), scale=0.6,
        color=(200, 200, 200), bg_color=(0, 0, 0), alpha=0.6
    )

    return frame


def main():
    parser = argparse.ArgumentParser(description="Real-time fruit freshness detection")
    parser.add_argument("--model", default="best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    args = parser.parse_args()

    # Check model file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please train first with: python baseline.py")
        sys.exit(1)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path}...")
    try:
        model, ckpt = load_model_from_checkpoint(model_path, map_location=device)
        model = model.to(device)
        class_names = ckpt["class_names"]
        print(f"Model loaded. Classes: {class_names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Get transforms (224x224 is standard for ResNet)
    transform = get_val_transform(image_size=224)

    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        print("Available cameras:")
        for i in range(5):
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                print(f"  Camera {i}: Available")
                test_cap.release()
        sys.exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")

    print("\n" + "="*50)
    print("SpoiledOrNot - Real-time Fruit Freshness Detector")
    print("="*50)
    print("Instructions:")
    print("  - Show a fruit to the camera")
    print("  - Press SPACE to capture and classify")
    print("  - Press ESC or Q to quit")
    print("="*50 + "\n")

    last_prediction = None
    last_confidence = 0.0
    last_probs = [0.0] * len(class_names)
    freeze_frame = None
    is_frozen = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        display_frame = frame.copy()

        if is_frozen and freeze_frame is not None:
            # Show the captured frame with results
            display_frame = freeze_frame.copy()
            display_frame = draw_prediction(
                display_frame, last_prediction, last_confidence,
                class_names, last_probs, capture_mode=True
            )
        else:
            # Live preview - draw a guide box
            h, w = display_frame.shape[:2]
            box_size = min(h, w) // 2
            x1, y1 = (w - box_size) // 2, (h - box_size) // 2
            x2, y2 = x1 + box_size, y1 + box_size

            # Draw center guide box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Position fruit here",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            display_frame = draw_prediction(
                display_frame, None, 0, class_names, [], capture_mode=False
            )

        # Show the frame
        cv2.imshow("SpoiledOrNot - Fruit Freshness Detector", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):  # ESC or Q
            break
        elif key == ord(' '):  # SPACE - capture
            if not is_frozen:
                # Capture from center region
                h, w = frame.shape[:2]
                box_size = min(h, w) // 2
                x1, y1 = (w - box_size) // 2, (h - box_size) // 2
                x2, y2 = x1 + box_size, y1 + box_size
                crop = frame[y1:y2, x1:x2]

                # Classify
                print("\n--- Capturing and classifying... ---")
                last_prediction, last_confidence, last_probs = classify_frame(
                    crop, model, transform, class_names, device
                )
                print(f"Result: {last_prediction} ({last_confidence*100:.1f}% confidence)")
                for cls, prob in zip(class_names, last_probs):
                    print(f"  {cls}: {prob*100:.1f}%")
                print("--- Press SPACE to capture again, or ESC to continue ---\n")

                freeze_frame = frame.copy()
                is_frozen = True
            else:
                # Resume live preview
                is_frozen = False
                freeze_frame = None

    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed. Goodbye!")


if __name__ == "__main__":
    main()
