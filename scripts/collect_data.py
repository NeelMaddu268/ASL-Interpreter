# scripts/collect_data.py

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from datetime import datetime

LABEL = input("Enter the label for this gesture (e.g., A, B, C, None): ")
SAVE_DIR = "data/processed_features"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sample_count = 0

print("[INFO] Press 's' to save a sample, 'q' to quit.")

# ====== FEATURE ENGINEERING FUNCTION ======
def extract_normalized_features(landmarks):
    # Reference point: Wrist (landmark index 0)
    wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
    
    features = []
    
    for i in range(1, 21):  # Exclude wrist itself
        point = np.array([landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z])
        
        # Calculate Euclidean distance from wrist
        dist = np.linalg.norm(point - wrist)
        
        # Normalize by hand size (distance from wrist to middle fingertip)
        mid_finger_tip = np.array([landmarks.landmark[12].x, landmarks.landmark[12].y, landmarks.landmark[12].z])  # index 12 = middle fingertip
        norm_factor = np.linalg.norm(mid_finger_tip - wrist) + 1e-6  # avoid division by zero
        
        normalized_dist = dist / norm_factor
        features.append(normalized_dist)

    return features  # Now you have 20 normalized distance features
# ==========================================

while True:
    success, frame = cap.read()
    if not success:
        continue

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(image, f"Label: {LABEL}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Landmark Capture", image)

    key = cv2.waitKey(1)

    if key == ord('s') and result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]

        # ====== Use new features ======
        features = extract_normalized_features(landmarks)
        features.append(LABEL)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = f"{LABEL}_{timestamp}.csv"
        filepath = os.path.join(SAVE_DIR, filename)

        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(features)

        sample_count += 1
        print(f"[SAVED] Sample {sample_count} saved as {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
