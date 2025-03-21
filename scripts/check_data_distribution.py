# scripts/check_data_distribution.py

import os
import csv
from collections import Counter

DATA_DIR = "data/processed_features"
label_counts = Counter()

for label_folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, label_folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row:
                            label_counts[label_folder] += 1

print("\n📊 Sample Count Per Class:")
for label, count in sorted(label_counts.items()):
    print(f"{label}: {count} samples")

total = sum(label_counts.values())
print(f"\nTotal samples: {total}")
