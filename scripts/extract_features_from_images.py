# scripts/extract_features_from_images.py

import os
import csv
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

INPUT_DIR = "datasets/train"  # your dataset folder
OUTPUT_DIR = "data/processed_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# ===== Feature extractor (same as live interpreter) =====
def extract_normalized_features(landmarks):
    wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
    mid_finger_tip = np.array([landmarks.landmark[12].x, landmarks.landmark[12].y, landmarks.landmark[12].z])
    norm_factor = np.linalg.norm(mid_finger_tip - wrist) + 1e-6
    features = []
    for i in range(1, 21):
        point = np.array([landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z])
        dist = np.linalg.norm(point - wrist)
        normalized_dist = dist / norm_factor
        features.append(normalized_dist)
    return features

# ===== Count total images first (for progress bar) =====
total_images = sum(
    len([f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    for _, _, files in os.walk(INPUT_DIR)
)

print(f"[INFO] Processing {total_images} images with MediaPipe...")

processed, skipped = 0, 0

with tqdm(total=total_images, desc="Extracting Features") as pbar:
    for label in os.listdir(INPUT_DIR):
        class_folder = os.path.join(INPUT_DIR, label)
        if not os.path.isdir(class_folder):
            continue

        output_folder = os.path.join(OUTPUT_DIR, label)
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(class_folder):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                pbar.update(1)
                continue

            image_path = os.path.join(class_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                pbar.update(1)
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                features = extract_normalized_features(result.multi_hand_landmarks[0])
                features.append(label)

                csv_name = filename.replace('.jpg', '.csv').replace('.png', '.csv')
                output_path = os.path.join(output_folder, csv_name)

                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(features)

                processed += 1
            else:
                skipped += 1

            pbar.update(1)

print(f"\n✅ Done. {processed} samples extracted | {skipped} skipped (no hand detected)")
