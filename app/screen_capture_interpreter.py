# app/screen_capture_interpreter.py

import mss
import numpy as np
import cv2
import mediapipe as mp
import joblib
from collections import deque, Counter

# Load model
clf = joblib.load("models/static_sign_classifier.pkl")
labels = clf.classes_

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Feature extraction (same as real-time interpreter)
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

# Initialize smoothing buffer
prediction_history = deque(maxlen=10)

# Screen capture setup
sct = mss.mss()

# Choose the monitor or region to capture
monitor = sct.monitors[1]  # Full primary screen
# Optional: specify region like {'top':100, 'left':100, 'width':640, 'height':480}

print("[INFO] Screen Capture Interpreter running — press 'q' to quit")

while True:
    # Capture screen
    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    predicted_label = "No Hand"
    confidence = 0.0

    # Process frame through MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_normalized_features(hand_landmarks)
            probs = clf.predict_proba(features)[0]
            max_idx = np.argmax(probs)
            raw_prediction = labels[max_idx]
            confidence = probs[max_idx]

            prediction_history.append(raw_prediction)
            predicted_label = Counter(prediction_history).most_common(1)[0][0]

    # Show prediction + confidence
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.putText(frame, f"Confidence: {confidence*100:.2f}%", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = 10, 100, 300, 20
    filled_w = int(confidence * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)

    # Display screen capture with predictions
    cv2.imshow("Screen Capture Sign Interpreter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
