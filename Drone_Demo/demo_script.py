import os
from ultralytics import YOLO
import cv2

# --- Configuration ---
# Path to model weights.
model_path = r"C:\Users\rache\Desktop\תואר מדעי המחשב\AI project\drone-for-your-safety-GenAI\runs\detect\sar_demo_model3\weights\best.pt"
images_folder = 'test_images'

print(f"Loading model from: {model_path}...")
try:
    model = YOLO(model_path)
except Exception as e:
    print("Error: Could not load model. Make sure the path is correct.")
    print(f"System message: {e}")
    exit()

print("\nStarting Demo Runs...")

# Retrieve all supported image files from the target directory
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

if not image_files:
    print("No images found in 'test_images' folder!")
    exit()

for i, img_name in enumerate(image_files, 1):
    img_path = os.path.join(images_folder, img_name)
    print(f"\n[{i}/{len(image_files)}] Processing: {img_name}")

    # Run inference pipeline
    # conf=0.25: Minimum confidence threshold for detections
    # save=True: Persist annotated images to disk
    results = model.predict(source=img_path, save=True, conf=0.15)

    # Process and log detection results
    for result in results:
        boxes = result.boxes
        print(f"   -> Detected {len(boxes)} objects.")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            print(f"      - {class_name} (Confidence: {conf:.2f})")

print("\n--- Demo Completed Successfully! ---")
print("Check the 'runs/detect' folder to see the output images.")