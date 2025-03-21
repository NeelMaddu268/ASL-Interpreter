# app/real_time_interpreter.py

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import sys
import os


print(sys.path)


# Force include root dir in Python path (absolute fix)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print(sys.path)

from utils.tts_speaker import speak


MUTE_TTS = True  # Set to True to disable TTS (toggle manually)


# ===== STEP 1: Load your trained modeal =====
model_path = "models/static_sign_classifier.pkl"
clf = joblib.load(model_path)
labels = clf.classes_

# ===== STEP 2: Initialize MediaPipe Hands =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ===== STEP 3: Helper function to extract normalized features =====
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
    return np.array(features).reshape(1, -1)

# ===== STEP 4: Setup webcam and prediction buffer =====
cap = cv2.VideoCapture(0)
prediction_history = deque(maxlen=10)  # for smoothing

print("[INFO] Real-Time Interpreter running. Press 'q' to quit.")

# ===== STEP 5: Real-time loop =====
prev_prediction = ""
while True:
    success, frame = cap.read()
    if not success:
        continue

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_label = "No Hand"
    confidence = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_normalized_features(hand_landmarks)
            probs = clf.predict_proba(features)[0]
            max_idx = np.argmax(probs)
            raw_prediction = labels[max_idx]
            confidence = probs[max_idx]

            # Add raw prediction to smoothing buffer
            prediction_history.append(raw_prediction)
            # Smoothed output = most frequent in buffer
            predicted_label = Counter(prediction_history).most_common(1)[0][0]

            # Only speak if prediction has changed AND is not "No Hand"
            if not MUTE_TTS and predicted_label != "No Hand" and predicted_label != prev_prediction:
                speak(predicted_label)
                prev_prediction = predicted_label


    # ===== Display predicted label and confidence =====
    cv2.putText(image, f"Predicted: {predicted_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.putText(image, f"Confidence: {confidence*100:.2f}%", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ===== Draw confidence bar =====
    bar_x, bar_y, bar_w, bar_h = 10, 100, 300, 20
    filled_w = int(confidence * bar_w)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)

    # ===== Show live window =====
    cv2.imshow("Real-Time Sign Language Interpreter", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== Cleanup =====
cap.release()
cv2.destroyAllWindows()
