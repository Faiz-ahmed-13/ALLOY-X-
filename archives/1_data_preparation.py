import os
from pathlib import Path

base_path = Path(r"C:\Users\Faiz Ahmed\OneDrive\Desktop\aluminum\PROJECT-1\dataset")
for item in base_path.iterdir():
    print(item.name)