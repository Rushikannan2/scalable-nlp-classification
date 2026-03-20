import pandas as pd
import joblib
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load dataset
df = pd.read_csv(r"D:\IIT-Gandhinagar_Project\sample_100k.csv")

X = df["DATA"]
y = df["TOPIC"]

# Train / Validation / Test split (70 / 10 / 20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=SEED, stratify=y_temp
)

# TF-IDF (improved settings)
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# SVM Model
model = LinearSVC(random_state=SEED)
model.fit(X_train_vec, y_train)

# Train evaluation
train_pred = model.predict(X_train_vec)
train_report = classification_report(y_train, train_pred)

print("\n========== TRAIN RESULTS ==========\n")
print(train_report)

# Validation evaluation
val_pred = model.predict(X_val_vec)
val_report = classification_report(y_val, val_pred)

print("\n========== VALIDATION RESULTS ==========\n")
print(val_report)

# Test evaluation
test_pred = model.predict(X_test_vec)
test_report = classification_report(y_test, test_pred)

print("\n========== TEST RESULTS ==========\n")
print(test_report)

# Save results (clean formatting)
results_path = r"D:\IIT-Gandhinagar_Project\experiments\ml_results.txt"

with open(results_path, "a") as f:
    f.write("\n" + "="*80 + "\n")
    f.write("SVM (LinearSVC) RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("TRAIN RESULTS:\n")
    f.write(train_report + "\n\n")

    f.write("VALIDATION RESULTS:\n")
    f.write(val_report + "\n\n")

    f.write("TEST RESULTS:\n")
    f.write(test_report + "\n")

# Save model and vectorizer
model_path = r"D:\IIT-Gandhinagar_Project\final_models\svm_model.pkl"
vectorizer_path = r"D:\IIT-Gandhinagar_Project\final_models\tfidf_vectorizer_svm.pkl"

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("\nSaved:")
print("Model ->", model_path)
print("Vectorizer ->", vectorizer_path)