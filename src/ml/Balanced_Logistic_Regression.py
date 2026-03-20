import pandas as pd
import numpy as np
import re
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Reproducibility

SEED = 42
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
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'\d+', '', text)              # remove numbers
    text = re.sub(r'[^a-z\s]', '', text)         # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()     # clean spaces
    return text

# Load Data

df = pd.read_csv(DATA_PATH)

X = df["DATA"].astype(str).apply(preprocess)
y = df["TOPIC"]


# Train / Val / Test Split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=SEED, stratify=y_temp
)

# TF-IDF Vectorizer (IMPROVED)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),   # bigrams boost performance
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Model (Balanced Logistic Regression)

model = LogisticRegression(
    max_iter=300,
    class_weight='balanced',
    n_jobs=-1
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


# Reports

train_report = evaluate("TRAIN", y_train, train_pred)
val_report = evaluate("VALIDATION", y_val, val_pred)
test_report = evaluate("TEST", y_test, test_pred)

# Save Results (APPEND MODE)

with open(RESULT_PATH, "a") as f:
    f.write("\n" + "="*80 + "\n")
    f.write("BALANCED LOGISTIC REGRESSION (FINAL)\n")
    f.write("="*80 + "\n\n")

    f.write("TRAIN RESULTS:\n")
    f.write(train_report + "\n\n")

    f.write("VALIDATION RESULTS:\n")
    f.write(val_report + "\n\n")

    f.write("TEST RESULTS:\n")
    f.write(test_report + "\n")

# Save Model + Vectorizer

joblib.dump(model, os.path.join(MODEL_PATH, "balanced_logistic_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_PATH, "tfidf_vectorizer_balanced.pkl"))

print("\n Balanced Logistic Regression saved successfully!")