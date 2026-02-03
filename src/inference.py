import cv2
import os
import glob
import torch
import argparse
from ultralytics import YOLO


def run_inference(model_path, source_dir, output_dir, conf=0.3):
    """
    Runs YOLOv11 inference. Draws Fire/Smoke (background) and People (foreground).
    """
    # Load model
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(source_dir, ext)))

    if not images:
        print(f"No images found in {source_dir}")
        return

    print(f"Processing {len(images)} images...")

    for img_path in images:
        filename = os.path.basename(img_path)

        # Run inference with agnostic NMS to handle overlapping classes
        results = model.predict(img_path, conf=conf, iou=0.5, agnostic_nms=True, verbose=False)
        result = results[0]

        img_out = result.orig_img.copy()
        boxes = result.boxes

        # Separate classes
        people = boxes[boxes.cls == 0]  # Class 0: Person
        hazards = boxes[boxes.cls != 0]  # Class 1,2: Fire/Smoke

        # Layer 1: Draw Hazards (Background)
        # Sort by confidence and take top 5 to reduce clutter
        if len(hazards) > 0:
            top_hazards = hazards[torch.argsort(hazards.conf, descending=True)[:5]]
            for box in top_hazards:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])

                # Red for Fire (1), Orange for Smoke (2)
                color = (0, 0, 255) if cls_id == 1 else (0, 140, 255)
                label = "Fire" if cls_id == 1 else "Smoke"

                cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Layer 2: Draw People (Foreground - Always on Top)
        for box in people:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_val = float(box.conf[0])

            # Bright Green, Thicker Line
            color = (0, 255, 0)
            cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 3)

            label_text = f"PERSON {conf_val:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Label background for readability
            cv2.rectangle(img_out, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(img_out, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Save result
        cv2.imwrite(os.path.join(output_dir, f"pred_{filename}"), img_out)

    print(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    # Default paths (relative to project root)
    # Update these or use command line args
    MODEL_PATH = "models/best.pt"
    INPUT_DIR = "dataset/test/images"
    OUTPUT_DIR = "results/inference_output"

    if os.path.exists(MODEL_PATH):
        run_inference(MODEL_PATH, INPUT_DIR, OUTPUT_DIR)
    else:
        print("Model file not found.")