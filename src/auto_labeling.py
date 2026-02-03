import cv2
import os
import glob
import numpy as np


def generate_hsv_labels(source_dir, output_dir):
    """
    Scans images and generates Fire/Smoke labels using HSV color thresholding.
    """
    # Class IDs (Must match data.yaml)
    ID_FIRE = 1
    ID_SMOKE = 2

    # Setup output structure
    images_out = os.path.join(output_dir, 'images')
    labels_out = os.path.join(output_dir, 'labels')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # Collect images
    extensions = ['*.jpg', '*.png', '*.jpeg']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(source_dir, ext)))

    print(f"Starting auto-labeling on {len(images)} images...")
    count_labeled = 0

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None: continue

        # Save copy of image
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(images_out, filename), img)

        h, w, _ = img.shape

        # Pre-processing: Blur to reduce noise
        blurred_img = cv2.GaussianBlur(img, (25, 25), 0)
        hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

        new_labels = []
        found_any = False

        # --- Step 1: Detect Fire (Orange/Yellow) ---
        lower_f1 = np.array([18, 50, 100]);
        upper_f1 = np.array([35, 255, 255])
        lower_f2 = np.array([0, 150, 100]);
        upper_f2 = np.array([10, 255, 255])

        mask_fire = cv2.bitwise_or(cv2.inRange(hsv, lower_f1, upper_f1),
                                   cv2.inRange(hsv, lower_f2, upper_f2))

        # Dilate to merge close regions
        kernel_fire = np.ones((15, 15), np.uint8)
        mask_fire = cv2.dilate(mask_fire, kernel_fire, iterations=2)

        contours_fire, _ = cv2.findContours(mask_fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_fire:
            area = cv2.contourArea(cnt)
            # Filter by area (ignore small noise or full-screen glitches)
            if area < 500 or area > (h * w * 0.9): continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            # Filter by aspect ratio (ignore thin lines)
            if cw / float(ch) < 0.2: continue

            found_any = True
            # Format: class x_center y_center width height (Normalized)
            new_labels.append(f"{ID_FIRE} {(x + cw / 2) / w:.6f} {(y + ch / 2) / h:.6f} {cw / w:.6f} {ch / h:.6f}\n")

        # --- Step 2: Detect Smoke (White/Gray) ---
        lower_s1 = np.array([0, 0, 120]);
        upper_s1 = np.array([180, 40, 255])
        mask_smoke = cv2.inRange(hsv, lower_s1, upper_s1)

        # Subtract fire mask to avoid overlap
        mask_smoke = cv2.subtract(mask_smoke, mask_fire)

        # Strong dilation for smoke
        kernel_smoke = np.ones((40, 40), np.uint8)
        mask_smoke = cv2.dilate(mask_smoke, kernel_smoke, iterations=3)

        contours_smoke, _ = cv2.findContours(mask_smoke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_smoke:
            area = cv2.contourArea(cnt)
            if area < 2000 or area > (h * w * 0.9): continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            found_any = True
            new_labels.append(f"{ID_SMOKE} {(x + cw / 2) / w:.6f} {(y + ch / 2) / h:.6f} {cw / w:.6f} {ch / h:.6f}\n")

        # Save label file if detections found
        if found_any:
            txt_name = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(labels_out, txt_name), 'w') as f:
                f.writelines(new_labels)
            count_labeled += 1

    print(f"Auto-labeling complete. Processed {count_labeled} images.")


if __name__ == "__main__":
    # Example usage
    # generate_hsv_labels("raw_images", "labeled_dataset")
    pass