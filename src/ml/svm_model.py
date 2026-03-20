import pandas as pd
import numpy as np
import re
import os
import joblib
import random
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Reproducibility

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# Paths

BASE_PATH = r"D:\IIT-Gandhinagar_Project"
DATA_PATH = os.path.join(BASE_PATH, "sample_100k.csv")
MODEL_PATH = os.path.join(BASE_PATH, "final_models")
RESULT_PATH = os.path.join(BASE_PATH, "experiments", "ml_results.txt")

os.makedirs(MODEL_PATH, exist_ok=True)


# Preprocessing

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load Data

df = pd.read_csv(DATA_PATH)

X = df["DATA"].astype(str).apply(preprocess)
y = df["TOPIC"]


# Label Encoding (IMPORTANT SAVE)

labels = {label: idx for idx, label in enumerate(sorted(y.unique()))}
y_encoded = y.map(labels)

# Train / Val / Test Split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=SEED, stratify=y_temp
)

# TF-IDF Vectorizer

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Model

model = LinearSVC(
    random_state=SEED,
    max_iter=2000
)

model.fit(X_train_vec, y_train)


# Evaluation Function

def evaluate(name, y_true, y_pred):
    print(f"\n========== {name} RESULTS ==========\n")
    report = classification_report(y_true, y_pred)
    print(report)
    return report


# Predictions

train_pred = model.predict(X_train_vec)
val_pred = model.predict(X_val_vec)
test_pred = model.predict(X_test_vec)

train_report = evaluate("TRAIN", y_train, train_pred)
val_report = evaluate("VALIDATION", y_val, val_pred)
test_report = evaluate("TEST", y_test, test_pred)


# Save Results

with open(RESULT_PATH, "a") as f:
    f.write("\n" + "="*80 + "\n")
    f.write("SVM (LinearSVC) FINAL\n")
    f.write("="*80 + "\n\n")

    f.write("TRAIN RESULTS:\n")
    f.write(train_report + "\n\n")

    f.write("VALIDATION RESULTS:\n")
    f.write(val_report + "\n\n")

    f.write("TEST RESULTS:\n")
    f.write(test_report + "\n")


# SAVE EVERYTHING (IMPORTANT )


# 1. Model
joblib.dump(model, os.path.join(MODEL_PATH, "svm_model.pkl"))

# 2. Vectorizer
joblib.dump(vectorizer, os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))

# 3. Label Mapping
joblib.dump(labels, os.path.join(MODEL_PATH, "label_map.pkl"))

# 4. Reverse Label Mapping (VERY USEFUL)
inv_labels = {v: k for k, v in labels.items()}
joblib.dump(inv_labels, os.path.join(MODEL_PATH, "inverse_label_map.pkl"))

# 5. Preprocessing Description
joblib.dump({
    "steps": [
        "lowercase",
        "remove_urls",
        "remove_numbers",
        "remove_special_characters",
        "normalize_spaces"
    ]
}, os.path.join(MODEL_PATH, "preprocessing.pkl"))

# 6. Config (FULL REPRODUCIBILITY)
config = {
    "model": "LinearSVC",
    "max_iter": 2000,
    "random_state": SEED,
    "vectorizer": {
        "max_features": 20000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95
    },
    "data": {
        "train_split": 0.7,
        "val_split": 0.1,
        "test_split": 0.2
    }
}

joblib.dump(config, os.path.join(MODEL_PATH, "config.pkl"))

# 7. Label Distribution (for report/debug)
joblib.dump(y.value_counts().to_dict(),
            os.path.join(MODEL_PATH, "label_distribution.pkl"))

print("\n EVERYTHING saved successfully (MODEL + CONFIG + PREPROCESSING + LABELS)")