import os
import random
from PIL import Image
from pathlib import Path

# --- Configuration ---

BASE_DIR = Path(__file__).resolve().parent
INPUT_BASE_FOLDER = 'input_images_test'
OUTPUT_BASE_FOLDER = BASE_DIR.parent / "Dataset_Fire_Generetor_Result" / "test"
FIRE_ASSETS_FOLDER = 'fire_images'

FIRE_CLASS_ID = 12
FIRE_PROBABILITY = 1.0

# Orange Tint Settings
TINT_COLOR = (255, 80, 0)
TINT_OPACITY = 30

# Use an empty string because there are no subfolders like 'valid'
SUBSETS = ['']

# --- Main Processing ---

count = 0

# Check if fire assets exist
if not os.path.exists(FIRE_ASSETS_FOLDER):
    print(f"Error: Folder {FIRE_ASSETS_FOLDER} not found!")
    exit()

fire_images = [f for f in os.listdir(FIRE_ASSETS_FOLDER) if f.endswith('.png')]
if not fire_images:
    print("Error: No PNG images found in fire_assets folder!")
    exit()

print(f"Starting process on {INPUT_BASE_FOLDER}...")

for subset in SUBSETS:
    # Path is now just INPUT_BASE_FOLDER
    subset_path = os.path.join(INPUT_BASE_FOLDER, subset)

    if not os.path.exists(subset_path):
        print(f"Error: Could not find folder: {subset_path}")
        continue

    # Create standard YOLO output structure
    out_img_path = os.path.join(OUTPUT_BASE_FOLDER, 'images')
    out_lbl_path = os.path.join(OUTPUT_BASE_FOLDER, 'labels')

    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_lbl_path, exist_ok=True)

    for filename in os.listdir(subset_path):
        # Process only images
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 1. Load Background
        bg_full_path = os.path.join(subset_path, filename)
        try:
            background = Image.open(bg_full_path).convert("RGBA")
        except:
            continue

        bg_w, bg_h = background.size
        should_add_fire = random.random() < FIRE_PROBABILITY
        new_fire_line = ""

        if should_add_fire:
            # 2. Add Fire
            fire_name = random.choice(fire_images)
            fire = Image.open(os.path.join(FIRE_ASSETS_FOLDER, fire_name)).convert("RGBA")

            # Scale: 10% to 30% of background width
            scale_factor = random.uniform(0.1, 0.3)
            new_fire_w = int(bg_w * scale_factor)
            aspect_ratio = fire.height / fire.width
            new_fire_h = int(new_fire_w * aspect_ratio)
            fire = fire.resize((new_fire_w, new_fire_h), Image.Resampling.LANCZOS)

            max_x = bg_w - new_fire_w
            max_y = bg_h - new_fire_h

            # Position: Ground level (bottom 30% of image)
            paste_x = random.randint(0, max(0, max_x))
            paste_y = random.randint(int(max_y * 0.7), max(int(max_y * 0.7), max_y))

            # Paste using transparency mask
            background.paste(fire, (paste_x, paste_y), fire)

            # 3. Add Tint
            tint_layer = Image.new('RGBA', background.size, TINT_COLOR + (TINT_OPACITY,))
            background = Image.alpha_composite(background, tint_layer)

            # YOLO Label calculation
            center_x = (paste_x + (new_fire_w / 2)) / bg_w
            center_y = (paste_y + (new_fire_h / 2)) / bg_h
            norm_width = new_fire_w / bg_w
            norm_height = new_fire_h / bg_h
            new_fire_line = f"{FIRE_CLASS_ID} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n"

        # 4. Save Image
        final_image = background.convert("RGB")
        final_image.save(os.path.join(out_img_path, filename))

        # 5. Merge Labels
        src_txt_path = os.path.join(subset_path, os.path.splitext(filename)[0] + ".txt")
        existing_labels = []
        if os.path.exists(src_txt_path):
            with open(src_txt_path, "r") as f:
                existing_labels = f.readlines()

        dst_txt_path = os.path.join(out_lbl_path, os.path.splitext(filename)[0] + ".txt")
        with open(dst_txt_path, "w") as f:
            for line in existing_labels:
                f.write(line)
            if should_add_fire:
                f.write(new_fire_line)

        count += 1

print(f"Done! Processed {count} images. Results are in '{OUTPUT_BASE_FOLDER}'")