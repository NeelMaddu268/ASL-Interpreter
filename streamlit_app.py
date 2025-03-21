# streamlit_app.py

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque, Counter
import threading


clf = joblib.load("models/static_sign_classifier.pkl")

# Load model
clf = joblib.load("models/static_sign_classifier.pkl")
labels = clf.classes_

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Feature extractor
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

# UI Setup
st.set_page_config(page_title="ASL Interpreter", layout="centered")
st.title("🤟 Real-Time ASL Interpreter")
st.markdown("This app detects ASL letters using your webcam and a MediaPipe-powered ML model.")

run = st.checkbox("Start Webcam")
mute_tts = st.checkbox("Mute Text-to-Speech")

frame_display = st.empty()
prediction_text = st.empty()

prediction_history = deque(maxlen=10)
prev_prediction = ""

if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        predicted_label = "No Hand"
        confidence = 0.0

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = extract_normalized_features(hand_landmarks)
                features = np.array(features).reshape(1, -1)
                probs = clf.predict_proba(features)[0]
                max_idx = np.argmax(probs)
                predicted_label = labels[max_idx]
                confidence = probs[max_idx]
                prediction_history.append(predicted_label)

        if prediction_history:
            smoothed_prediction = Counter(prediction_history).most_common(1)[0][0]
        else:
            smoothed_prediction = "No Prediction"

        from utils.tts_speaker import speak

        if "last_spoken" not in st.session_state:
            st.session_state.last_spoken = ""

        def threaded_speak(text):
            tts_thread = threading.Thread(target=speak, args=(text,))
            tts_thread.start()

        if not mute_tts and smoothed_prediction != st.session_state.last_spoken and smoothed_prediction not in ["No Hand", "No Prediction"]:
            try:
                threaded_speak(smoothed_prediction)
                st.session_state.last_spoken = smoothed_prediction
            except Exception as e:
                st.warning(f"TTS failed: {e}")



        cv2.putText(frame, f"Prediction: {smoothed_prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence*100:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame_display.image(frame, channels="BGR")

        prediction_text.markdown(f"### ✅ Prediction: **{smoothed_prediction}** — {confidence*100:.2f}%")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
