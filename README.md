# 🤟 Real-Time ASL Sign Language Interpreter

A real-time ASL (American Sign Language) letter recognition system built using **MediaPipe**, **Scikit-Learn**, and **Streamlit**.

The app detects hand gestures from webcam input, classifies them into ASL letters, and optionally speaks them out loud using Text-to-Speech (TTS).

---

## 🚀 Live Demo (Streamlit Cloud)

👉 Add your Streamlit deployment link here after publishing your app.

---

## 📸 Features

- Real-time webcam-based ASL letter detection
- Trained on 59,000+ MediaPipe-extracted gesture samples
- Prediction smoothing with confidence scores
- Text-to-Speech output (with mute toggle)
- Model is auto-downloaded from Google Drive on first run

---

## 🧠 Tech Stack

- Python  
- OpenCV  
- MediaPipe  
- Scikit-Learn  
- Streamlit  
- gTTS  
- Pygame

---

## 📂 Project Structure
```
sign-lang-interpreter/
├── app/                         → Real-time interpreter scripts
├── models/                      → Model auto-downloads here
├── scripts/                     → Data collection and training scripts
├── utils/                       → TTS speaker module
├── requirements.txt
├── streamlit_app.py
└── README.md
```

---

## 📥 Local Setup Instructions

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/asl-sign-language-interpreter.git
cd asl-sign-language-interpreter
```

**2. Install required dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the app locally:**
```bash
streamlit run streamlit_app.py
```

---

## 📦 Model Download Info

The model file (`static_sign_classifier.pkl`) is **NOT included in this repo** (it exceeds GitHub's 100MB file limit).

Instead, the app automatically downloads it on first run from this Google Drive link:
https://drive.google.com/file/d/1t_hjYRqIMLI6WukQXEqQNQWFU7rMU0OA/view?usp=sharing

You don’t need to do anything manually — it’s handled in the code.

---

## ☁️ Streamlit Cloud Deployment

1. Push this project to your GitHub account
2. Go to https://streamlit.io/cloud
3. Click “Deploy an app”
4. Select your GitHub repo
5. Set `streamlit_app.py` as the entry point
6. Your app will launch and auto-download the model

---

## 🙌 Credits

- ASL Alphabet Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

---

## 🪪 License

MIT License — feel free to use, fork, and improve this project.
