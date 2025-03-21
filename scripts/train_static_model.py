# scripts/train_static_model.py

import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

# ===== STEP 1: Load processed normalized feature data =====
DATA_DIR = "data/processed_features"
#DATA_DIR = "datasets/train"


X = []
y = []

# Count total CSV files first for accurate progress bar
total_files = sum(
    len(files)
    for _, _, files in os.walk(DATA_DIR)
    if files
)

print(f"\n[INFO] Loading {total_files} feature files...")

with tqdm(total=total_files, desc="Loading Samples") as pbar:
    for label in os.listdir(DATA_DIR):
        label_folder = os.path.join(DATA_DIR, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                if filename.endswith(".csv"):
                    filepath = os.path.join(label_folder, filename)
                    with open(filepath, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if row:
                                features = list(map(float, row[:-1]))
                                X.append(features)
                                y.append(label)
                    pbar.update(1)


X = np.array(X)
y = np.array(y)

# ===== STEP 2: Train/Test Split =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== STEP 3: Train Random Forest Classifier =====
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ===== STEP 4: Evaluate Model =====
y_pred = clf.predict(X_test)

print("\n=== Accuracy Report ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# ===== STEP 5: Save Model =====
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(clf, os.path.join(MODEL_DIR, "static_sign_classifier.pkl"), compress=3, protocol=4)
print(f"\n✅ Model saved to {MODEL_DIR}/static_sign_classifier.pkl")
