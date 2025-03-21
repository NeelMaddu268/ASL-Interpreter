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


## 🙌 Credits

- ASL Alphabet Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

---

## 🪪 License

MIT License — feel free to use, fork, and improve this project.
