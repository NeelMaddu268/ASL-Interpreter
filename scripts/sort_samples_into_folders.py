# scripts/sort_samples_into_folders.py

import os
import shutil

DATA_DIR = "data/processed_features"
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".csv"):
        label = filename.split("_")[0]  # 'A_202503...' → 'A'
        label_folder = os.path.join(DATA_DIR, label)
        os.makedirs(label_folder, exist_ok=True)

        src = os.path.join(DATA_DIR, filename)
        dst = os.path.join(label_folder, filename)
        shutil.move(src, dst)

print("✅ Samples sorted into label folders.")
