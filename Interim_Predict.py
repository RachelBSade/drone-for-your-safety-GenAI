from ultralytics import YOLO
import os
from pathlib import Path


def run_detection_report():
    # Paths configuration
    base_dir = Path(__file__).resolve().parent
    weights_path = base_dir / "runs" / "detect" / "Interim" / "fire_detection_demo" / "weights" / "best.pt"
    images_dir = base_dir / "Dataset_Fire_Generetor_Result" / "val" / "images"

    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    # Load model
    model = YOLO(weights_path)

    # Run inference
    # stream=True helps managing memory for large datasets
    results = model.predict(
        source=images_dir,
        save=True,
        conf=0.2,
        device='cpu',
        verbose=False  # Reduces console clutter
    )

    # Summary report logic
    print(f"{'Image Name':<40} | {'Objects Detected':<20}")
    print("-" * 65)

    total_objects = 0
    images_with_detections = 0

    for result in results:
        file_name = os.path.basename(result.path)
        count = len(result.boxes)  # Number of bounding boxes detected

        if count > 0:
            images_with_detections += 1
            total_objects += count

        print(f"{file_name:<40} | {count:<20}")

    # Final statistics
    print("-" * 65)
    print(f"Total images processed: {len(results)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total objects found: {total_objects}")


if __name__ == '__main__':
    run_detection_report()