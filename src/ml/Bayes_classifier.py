import pandas as pd
import numpy as np
import re
import os
import joblib
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# REPRODUCIBILITY

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# PATHS

BASE_PATH = r"D:\IIT-Gandhinagar_Project"
DATA_PATH = os.path.join(BASE_PATH, "sample_100k.csv")
MODEL_PATH = os.path.join(BASE_PATH, "final_models")
RESULT_PATH = os.path.join(BASE_PATH, "experiments", "ml_results.txt")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

# PREPROCESSING

def preprocess(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# LOAD DATA

df = pd.read_csv(DATA_PATH)

if "DATA" not in df.columns or "TOPIC" not in df.columns:
    raise ValueError("CSV must contain 'DATA' and 'TOPIC' columns")

X = df["DATA"].astype(str).apply(preprocess)
y = df["TOPIC"]


# LABEL ENCODING

labels = {label: idx for idx, label in enumerate(sorted(y.unique()))}
y_encoded = y.map(labels)

# SPLIT

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=SEED, stratify=y_temp
)

# IMPROVED TF-IDF

vectorizer = TfidfVectorizer(
    max_features=30000,              # more features
    ngram_range=(1,3),               # uni + bi + tri
    stop_words='english',            # remove common words
    min_df=2,                        # remove rare words
    max_df=0.9,                      # remove too frequent words
    sublinear_tf=True                # better scaling
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)


# HYPERPARAMETER TUNING

best_alpha = None
best_val_acc = 0
best_model = None

print("\n🔍 Tuning alpha...")

for alpha in [0.1, 0.3, 0.5, 1.0, 2.0]:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_vec, y_train)

    val_pred = model.predict(X_val_vec)
    acc = accuracy_score(y_val, val_pred)

    print(f"Alpha: {alpha} | Val Acc: {acc:.4f}")

    if acc > best_val_acc:
        best_val_acc = acc
        best_alpha = alpha
        best_model = model

print(f"\nBest Alpha: {best_alpha}")

model = best_model

# PARAMETER COUNT

num_features = X_train_vec.shape[1]
num_classes = len(labels)
total_params = num_features * num_classes

print("\n===== MODEL PARAMETERS =====")
print(f"Features: {num_features}")
print(f"Classes: {num_classes}")
print(f"Approx Parameters: {total_params:,}")


# EVALUATION

def evaluate(name, y_true, y_pred):
    print(f"\n========== {name} RESULTS ==========\n")

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

    return f"Accuracy: {acc:.4f}\n\n{report}"

train_pred = model.predict(X_train_vec)
val_pred = model.predict(X_val_vec)
test_pred = model.predict(X_test_vec)

train_report = evaluate("TRAIN", y_train, train_pred)
val_report = evaluate("VALIDATION", y_val, val_pred)
test_report = evaluate("TEST", y_test, test_pred)


# SAVE RESULTS

with open(RESULT_PATH, "a") as f:
    f.write("\n" + "="*80 + "\n")
    f.write("NAIVE BAYES (BEST)\n")
    f.write("="*80 + "\n\n")

    f.write(f"Best Alpha: {best_alpha}\n")
    f.write(f"Features: {num_features}\n")
    f.write(f"Classes: {num_classes}\n")
    f.write(f"Approx Parameters: {total_params}\n\n")

    f.write("TRAIN RESULTS:\n" + train_report + "\n\n")
    f.write("VALIDATION RESULTS:\n" + val_report + "\n\n")
    f.write("TEST RESULTS:\n" + test_report + "\n")


# SAVE EVERYTHING

joblib.dump(model, os.path.join(MODEL_PATH, "nb_best_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_PATH, "nb_best_vectorizer.pkl"))
joblib.dump(labels, os.path.join(MODEL_PATH, "nb_label_map.pkl"))

inv_labels = {v: k for k, v in labels.items()}
joblib.dump(inv_labels, os.path.join(MODEL_PATH, "nb_inverse_labels.pkl"))

joblib.dump({
    "model": "NaiveBayes",
    "best_alpha": best_alpha,
    "features": num_features,
    "ngram_range": (1,3),
    "seed": SEED
}, os.path.join(MODEL_PATH, "nb_best_config.pkl"))

joblib.dump({
    "steps": [
        "lowercase",
        "remove_urls",
        "remove_numbers",
        "remove_special_characters",
        "normalize_spaces"
    ]
}, os.path.join(MODEL_PATH, "nb_preprocessing.pkl"))

joblib.dump(y.value_counts().to_dict(),
            os.path.join(MODEL_PATH, "nb_label_distribution.pkl"))

print("\n NAIVE BAYES (BEST): EVERYTHING SAVED SUCCESSFULLY")