import cv2
import numpy as np
import os
import random
import glob
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
FIRE_CLASS_ID = 1
TINT_INTENSITY = 0.10
TINT_COLOR = (0, 100, 255)  # Orange-ish in BGR


# ==========================================
# Helper Functions
# ==========================================

def load_yolo_labels(label_path, img_w, img_h):
    """ Reads YOLO format labels and converts to pixel coordinates. """
    boxes = []
    if not os.path.exists(label_path):
        return []

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_c, y_c, w, h = parts[1:]

                x1 = int((x_c - w / 2) * img_w)
                y1 = int((y_c - h / 2) * img_h)
                w_px = int(w * img_w)
                h_px = int(h * img_h)

                boxes.append({'cls': cls_id, 'x1': x1, 'y1': y1, 'w': w_px, 'h': h_px})
    return boxes


def normalize_bbox(x, y, w, h, img_w, img_h):
    """ Normalizes pixel coordinates back to YOLO format (0-1). """
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return np.clip([x_c, y_c, w_norm, h_norm], 0, 1)


def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """ Overlays a transparent PNG onto a background image safely. """
    bg_h, bg_w, _ = background.shape

    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size)

    ov_h, ov_w, _ = overlay.shape

    # Check if totally out of bounds
    if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
        return background, None

    # Calculate cropping coordinates
    bg_x_start = max(0, x)
    bg_y_start = max(0, y)
    bg_x_end = min(bg_w, x + ov_w)
    bg_y_end = min(bg_h, y + ov_h)

    h_crop = bg_y_end - bg_y_start
    w_crop = bg_x_end - bg_x_start

    if h_crop <= 0 or w_crop <= 0:
        return background, None

    ov_x_start = bg_x_start - x
    ov_y_start = bg_y_start - y
    ov_x_end = ov_x_start + w_crop
    ov_y_end = ov_y_start + h_crop

    overlay_cropped = overlay[ov_y_start:ov_y_end, ov_x_start:ov_x_end]
    background_roi = background[bg_y_start:bg_y_end, bg_x_start:bg_x_end]

    # Alpha blending
    alpha_mask = overlay_cropped[:, :, 3] / 255.0
    img_rgb = overlay_cropped[:, :, :3]

    for c in range(3):
        background_roi[:, :, c] = (1. - alpha_mask) * background_roi[:, :, c] + \
                                  alpha_mask * img_rgb[:, :, c]

    return background, (bg_x_start, bg_y_start, w_crop, h_crop)


def apply_orange_tint(image, intensity, color):
    """ Applies a global orange tint to simulate fire glow. """
    overlay = np.full(image.shape, color, dtype='uint8')
    return cv2.addWeighted(image, 1.0 - intensity, overlay, intensity, 0)


def create_procedural_smoke_layer(img_h, img_w, fire_boxes):
    """ Generates a synthetic smoke layer based on fire positions. """
    smoke_layer = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    if not fire_boxes: return smoke_layer

    smoke_gray_val = random.randint(55, 90)
    smoke_color = (smoke_gray_val, smoke_gray_val, smoke_gray_val)

    for (fx, fy, fw, fh) in fire_boxes:
        current_y = fy + int(fh * 0.4)
        center_x = fx + fw // 2
        current_radius = int(fw * 0.6)
        wind_bias = random.uniform(-0.8, 0.8)
        current_alpha = 210.0

        # Simulate smoke rising
        while current_y > -200:
            rise_factor = (img_h - current_y) / img_h
            noise = random.randint(-15, 15)
            wind_effect = int(wind_bias * 20 * rise_factor)
            center_x += noise + wind_effect
            current_radius = int(current_radius * 1.02) + 1

            # Draw main smoke puff
            cv2.circle(smoke_layer, (center_x, current_y), current_radius, smoke_color + (int(current_alpha),), -1)

            # Draw scattered puffs
            if random.random() > 0.4:
                offset_x = random.randint(-current_radius, current_radius)
                offset_y = random.randint(-int(current_radius / 2), int(current_radius / 2))
                scatter_radius = int(current_radius * 0.7)
                scatter_alpha = int(current_alpha * 0.7)
                cv2.circle(smoke_layer, (center_x + offset_x, current_y + offset_y), scatter_radius,
                           smoke_color + (scatter_alpha,), -1)

            step = int(current_radius * 0.4)
            current_y -= max(5, step)
            current_alpha = max(0, current_alpha * 0.96)
            if current_alpha < 10: break

    # Blur the smoke for realism
    smoke_layer_blurred = cv2.GaussianBlur(smoke_layer, (151, 151), 0)
    return smoke_layer_blurred


# ==========================================
# Main Processing Logic
# ==========================================

def process_single_image(img_path, label_path, save_img_path, save_lbl_path, fire_assets):
    """ Processes a single image: adds fire, smoke, tint, and updates labels. """
    img = cv2.imread(img_path)
    if img is None: return False
    img_h, img_w = img.shape[:2]

    # Load existing labels
    existing_boxes = load_yolo_labels(label_path, img_w, img_h)
    new_fire_boxes = []

    aug_img = img.copy()

    # 1. Strategy Selection
    strategy = random.choice(['left', 'right', 'center'])
    env_x_start, env_x_end = 0, img_w

    if strategy == 'left':
        env_x_end = int(img_w * 0.45)
    elif strategy == 'right':
        env_x_start = int(img_w * 0.55)
    elif strategy == 'center':
        env_x_start = int(img_w * 0.30)
        env_x_end = int(img_w * 0.70)

    # 2. Add Fire on People (Foreground)
    for box in existing_boxes:
        person_center_x = box['x1'] + (box['w'] // 2)
        is_in_fire_zone = False

        if strategy == 'left' and person_center_x < env_x_end:
            is_in_fire_zone = True
        elif strategy == 'right' and person_center_x > env_x_start:
            is_in_fire_zone = True
        elif strategy == 'center' and (env_x_start < person_center_x < env_x_end):
            is_in_fire_zone = True

        if not is_in_fire_zone: continue

        person_w, person_h = box['w'], box['h']
        if person_w <= 0 or person_h <= 0: continue

        # Coverage check omitted for brevity, adding flames
        num_flames = random.randint(8, 15)

        for _ in range(num_flames):
            asset = random.choice(fire_assets)
            scale = random.uniform(0.3, 0.7)
            new_w, new_h = int(box['w'] * scale), int(box['h'] * scale)
            if new_w < 5: continue

            px = random.randint(box['x1'] - int(new_w / 2), box['x1'] + box['w'])
            py = random.randint(box['y1'] + int(box['h'] * 0.2), box['y1'] + box['h'])

            aug_img, bbox = overlay_transparent(aug_img, asset, px, py, (new_w, new_h))
            if bbox:
                new_fire_boxes.append(bbox)

    # 3. Environmental Fire (Background)
    min_y = int(img_h * 0.5)
    max_y = img_h - 50
    num_bg_flames = random.randint(100, 150)

    for _ in range(num_bg_flames):
        asset = random.choice(fire_assets)
        scale = random.uniform(0.15, 0.4)
        new_h_f = int(img_h * scale)
        if new_h_f == 0: continue
        ratio = asset.shape[1] / asset.shape[0]
        new_w_f = int(new_h_f * ratio)

        if env_x_end - new_w_f <= env_x_start: continue
        px = random.randint(env_x_start, env_x_end - new_w_f)
        py = random.randint(min_y, max_y - int(new_h_f / 2))

        aug_img, bbox = overlay_transparent(aug_img, asset, px, py, (new_w_f, new_h_f))
        if bbox: new_fire_boxes.append(bbox)

    # 4. Small Scattered Flames
    num_scattered_flames = random.randint(40, 80)
    for _ in range(num_scattered_flames):
        asset = random.choice(fire_assets)
        scale = random.uniform(0.08, 0.15)
        new_h_f = int(img_h * scale)
        if new_h_f == 0: continue
        ratio = asset.shape[1] / asset.shape[0]
        new_w_f = int(new_h_f * ratio)

        if env_x_end - new_w_f <= env_x_start: continue
        px = random.randint(env_x_start - int(img_w * 0.1), env_x_end + int(img_w * 0.1))
        py = random.randint(min_y - int(img_h * 0.1), max_y)
        aug_img, bbox = overlay_transparent(aug_img, asset, px, py, (new_w_f, new_h_f))
        if bbox: new_fire_boxes.append(bbox)

    # 5. Smoke Layer
    smoke_layer = create_procedural_smoke_layer(img_h, img_w, new_fire_boxes)
    smoke_alpha = smoke_layer[:, :, 3] / 255.0
    smoke_rgb = smoke_layer[:, :, :3]
    for c in range(3):
        aug_img[:, :, c] = (1.0 - smoke_alpha) * aug_img[:, :, c] + smoke_alpha * smoke_rgb[:, :, c]

    # 6. Tint
    aug_img = apply_orange_tint(aug_img, TINT_INTENSITY, TINT_COLOR)

    # 7. Save
    cv2.imwrite(save_img_path, aug_img)
    with open(save_lbl_path, 'w') as f:
        # Write existing people labels
        for box in existing_boxes:
            norm = normalize_bbox(box['x1'], box['y1'], box['w'], box['h'], img_w, img_h)
            f.write(f"{box['cls']} {' '.join(map(str, norm))}\n")
        # Write new fire labels
        for (fx, fy, fw, fh) in new_fire_boxes:
            norm = normalize_bbox(fx, fy, fw, fh, img_w, img_h)
            f.write(f"{FIRE_CLASS_ID} {' '.join(map(str, norm))}\n")

    return True


def augment_dataset(input_dir, output_dir, fire_assets_dir):
    """ Main driver function to augment entire dataset. """

    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    # Load fire assets
    fire_assets = []
    assets_paths = glob.glob(os.path.join(fire_assets_dir, "*.png"))
    for p in assets_paths:
        fire_assets.append(cv2.imread(p, cv2.IMREAD_UNCHANGED))

    if not fire_assets:
        print("Error: No fire assets found!")
        return

    # Get input images
    images = glob.glob(os.path.join(input_dir, 'images', '*.*'))
    print(f"Starting augmentation on {len(images)} images...")

    for img_path in tqdm(images):
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]

        label_path = os.path.join(input_dir, 'labels', name_no_ext + '.txt')
        save_img_path = os.path.join(output_dir, 'images', filename)
        save_lbl_path = os.path.join(output_dir, 'labels', name_no_ext + '.txt')

        process_single_image(img_path, label_path, save_img_path, save_lbl_path, fire_assets)

    print("Augmentation complete.")


if __name__ == "__main__":
    # Example usage (update paths before running)
    # augment_dataset("dataset/train", "dataset_augmented/train", "assets/fire")
    pass