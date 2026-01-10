import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR.parent / "Dataset_Fire_Generetor_Result"

DATA_SPLITS = {
    'train': DATASET_ROOT / "train" / "labels",
    'val':   DATASET_ROOT / "val" / "labels",
    'test':  DATASET_ROOT / "test" / "labels"
}

CLASS_NAMES = {
    0: 'awning-tricycle', 1: 'bicycle', 2: 'bus', 3: 'car',
    4: 'ignored regions', 5: 'motor', 6: 'others', 7: 'pedestrian',
    8: 'people', 9: 'tricycle', 10: 'truck', 11: 'van', 12: 'fire'
}


def load_split_data(split_name, folder_path):
    records = []
    files = glob.glob(os.path.join(folder_path, "*.txt"))

    for file_path in files:
        with open(file_path, 'r') as f:
            content = f.read()
            # Fix potentially merged lines for class 12 [cite: 2]
            content = content.replace('12 0.', '\n12 0.')
            lines = content.strip().split('\n')

            for line in lines:
                parts = line.strip().split()
                if not parts: continue

                try:
                    cls_id = int(parts[0])
                    # Standard YOLO format (5 parts) [cite: 5]
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                    # Polygon/OBB format (9+ parts) [cite: 6]
                    elif len(parts) >= 9:
                        cls = float(parts[0])
                        x_coords = [float(parts[i]) for i in range(1, len(parts), 2)][cite: 7]
                        y_coords = [float(parts[i]) for i in range(2, len(parts), 2)][cite: 7]
                        x = sum(x_coords) / len(x_coords)[cite: 8]
                        y = sum(y_coords) / len(y_coords)[cite: 8]
                        w = max(x_coords) - min(x_coords)[cite: 8]
                        h = max(y_coords) - min(y_coords)[cite: 8]
                    else:
                        continue

                    records.append({
                        'split': split_name,
                        'class_id': int(cls),
                        'class_name': CLASS_NAMES.get(int(cls), 'Unknown'),
                        'x_center': x,
                        'y_center': y,
                        'area': w * h
                    })

                except Exception:
                    continue
    return records


def plot_2d_heatmap(df, split='train', target_class=None):
    # Filter data by split and optionally by class
    subset = df[df['split'] == split]
    if target_class:
        subset = subset[subset['class_name'] == target_class]
        title = f'2D Heatmap: {target_class} ({split} set)'
    else:
        title = f'2D Heatmap: All Objects ({split} set)'

    plt.figure(figsize=(10, 8))

    # Create 2D histogram
    # Using 'hot' colormap to emphasize high-density areas
    plt.hist2d(subset['x_center'], subset['y_center'], bins=40, cmap='hot', range=[[0, 1], [0, 1]])

    plt.colorbar(label='Object Density')
    plt.title(title)
    plt.xlabel('Normalized X Center')
    plt.ylabel('Normalized Y Center')

    # Invert Y axis to match image coordinate system (top-left is 0,0)
    plt.gca().invert_yaxis()
    plt.show()


def run_spatial_eda():
    all_data = []
    for split, path in DATA_SPLITS.items():
        if os.path.exists(path):
            all_data.extend(load_split_data(split, path))

    if not all_data:
        print("No data loaded. Check paths and file formats.")
        return

    df = pd.DataFrame(all_data)

    # Generate general heatmap for training data
    plot_2d_heatmap(df, split='train')

    # Generate specific heatmap for the 'fire' class to check your generator's distribution
    if 'fire' in df['class_name'].values:
        plot_2d_heatmap(df, split='train', target_class='fire')


if __name__ == "__main__":
    run_spatial_eda()