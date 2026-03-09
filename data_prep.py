# EXECUTED_CODE/5_analyze_and_prepare_kolektor_corrected.py
import cv2
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_path = Path(r"C:\Users\Faiz Ahmed\OneDrive\Desktop\aluminum\PROJECT-1\dataset")
target_base_dir = Path("processed_data_kolektor_corrected")

def is_defective_label(label_path):
    """Check if a label image contains any defects (white pixels)"""
    try:
        label_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label_img is None:
            return False
        # Check if there are any white pixels (value 255) in the label
        return np.any(label_img == 255)
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return False

print("Analyzing KolektorSDD dataset for defects...")

# Collect all images and their labels
defective_images = []
non_defective_images = []

for folder in dataset_path.iterdir():
    if folder.is_dir():
        print(f"Processing {folder.name}...")
        
        # Get all part images (excluding label files)
        part_images = [f for f in folder.iterdir() if f.name.endswith('.jpg') and '_label' not in f.name]
        
        for part_img in part_images:
            # Find corresponding label file
            label_path = folder / f"{part_img.stem}_label.bmp"
            
            if label_path.exists():
                if is_defective_label(label_path):
                    defective_images.append(part_img)
                    # print(f"  {part_img.name}: DEFECTIVE")
                else:
                    non_defective_images.append(part_img)
                    # print(f"  {part_img.name}: non-defective")
            else:
                non_defective_images.append(part_img)
                # print(f"  {part_img.name}: non-defective (no label)")

print(f"\nAnalysis complete!")
print(f"Defective images: {len(defective_images)}")
print(f"Non-defective images: {len(non_defective_images)}")
print(f"Total images: {len(defective_images) + len(non_defective_images)}")

# Split data into train, validation, and test sets
defective_train, defective_temp = train_test_split(defective_images, test_size=0.3, random_state=42)
defective_valid, defective_test = train_test_split(defective_temp, test_size=0.5, random_state=42)

non_defective_train, non_defective_temp = train_test_split(non_defective_images, test_size=0.3, random_state=42)
non_defective_valid, non_defective_test = train_test_split(non_defective_temp, test_size=0.5, random_state=42)

print(f"\nSplitting data:")
print(f"Train: {len(defective_train)} defective, {len(non_defective_train)} non-defective")
print(f"Valid: {len(defective_valid)} defective, {len(non_defective_valid)} non-defective")
print(f"Test: {len(defective_test)} defective, {len(non_defective_test)} non-defective")

# Create target directories
splits = ['train', 'valid', 'test']
for split in splits:
    defective_dir = target_base_dir / split / "defective"
    non_defective_dir = target_base_dir / split / "non_defective"
    defective_dir.mkdir(parents=True, exist_ok=True)
    non_defective_dir.mkdir(parents=True, exist_ok=True)

# Copy images to target directories with unique names to avoid overwriting
def copy_images(image_list, target_dir):
    for i, img_path in enumerate(image_list):
        # Create unique filename to avoid overwriting
        unique_name = f"{img_path.parent.name}_{img_path.name}"
        shutil.copy2(img_path, target_dir / unique_name)

print("\nCopying images...")
copy_images(defective_train, target_base_dir / "train" / "defective")
copy_images(defective_valid, target_base_dir / "valid" / "defective")
copy_images(defective_test, target_base_dir / "test" / "defective")
copy_images(non_defective_train, target_base_dir / "train" / "non_defective")
copy_images(non_defective_valid, target_base_dir / "valid" / "non_defective")
copy_images(non_defective_test, target_base_dir / "test" / "non_defective")

print("Data preparation complete!")
print(f"Organized data saved to: {target_base_dir}")

# Verify the copy worked
print(f"\nVerifying copy:")
for split in splits:
    defective_dir = target_base_dir / split / "defective"
    non_defective_dir = target_base_dir / split / "non_defective"
    defective_count = len(list(defective_dir.glob('*.*')))
    non_defective_count = len(list(non_defective_dir.glob('*.*')))
    print(f"{split}: {defective_count} defective, {non_defective_count} non-defective")