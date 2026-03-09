# EXECUTED_CODE/4_explore_kolektor_structure.py
import os
from pathlib import Path

# Define the path to your extracted dataset
dataset_path = Path(r"C:\Users\Faiz Ahmed\OneDrive\Desktop\aluminum\PROJECT-1\dataset")

print("Exploring KolektorSDD structure in detail...")
print(f"Dataset path: {dataset_path}")

# Count total folders
folders = [f for f in dataset_path.iterdir() if f.is_dir()]
print(f"\nTotal sample folders: {len(folders)}")

# Explore the first few folders to understand the structure
defective_samples = 0
non_defective_samples = 0

print("\nExploring first 5 folders to understand structure:")
for i, folder in enumerate(folders[:5]):
    print(f"\n📁 {folder.name}:")
    files = list(folder.iterdir())
    for file in files:
        file_type = "IMAGE" if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] else "OTHER"
        print(f"  {file_type}: {file.name}")
    
    # Check if this folder contains defects (has annotation files)
    annotation_files = [f for f in files if f.suffix.lower() in ['.txt', '.json', '.xml'] or 'gt' in f.name.lower() or 'mask' in f.name.lower()]
    if annotation_files:
        defective_samples += 1
        print(f"  ⚠️  DEFECTIVE (has {len(annotation_files)} annotation files)")
    else:
        non_defective_samples += 1
        print(f"  ✅ NON-DEFECTIVE (no annotation files)")

print(f"\nBased on first 5 samples: {defective_samples} defective, {non_defective_samples} non-defective")

# Now let's check the entire dataset
print("\n\nAnalyzing entire dataset...")
defective_folders = []
non_defective_folders = []

for folder in folders:
    files = list(folder.iterdir())
    annotation_files = [f for f in files if f.suffix.lower() in ['.txt', '.json', '.xml'] or 'gt' in f.name.lower() or 'mask' in f.name.lower()]
    image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if annotation_files:
        defective_folders.append(folder.name)
    else:
        non_defective_folders.append(folder.name)

print(f"Defective samples: {len(defective_folders)}")
print(f"Non-defective samples: {len(non_defective_folders)}")
print(f"Total samples: {len(defective_folders) + len(non_defective_folders)}")

# Show a few examples
print(f"\nFirst 5 defective samples: {defective_folders[:5]}")
print(f"First 5 non-defective samples: {non_defective_folders[:5]}")