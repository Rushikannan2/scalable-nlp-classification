import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"D:\IIT-Gandhinagar_Project\sample_100k.csv")

X = df["DATA"]
y = df["TOPIC"]

# Train / Validation / Test split (70 / 10 / 20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=20000)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression with class balancing
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Train evaluation
train_pred = model.predict(X_train_vec)
train_report = classification_report(y_train, train_pred)

print("\nTrain Results:\n")
print(train_report)

# Validation evaluation
val_pred = model.predict(X_val_vec)
val_report = classification_report(y_val, val_pred)

print("\nValidation Results:\n")
print(val_report)

# Test evaluation
test_pred = model.predict(X_test_vec)
test_report = classification_report(y_test, test_pred)

print("\nTest Results:\n")
print(test_report)

# Save reports
with open(r"D:\IIT-Gandhinagar_Project\experiments\ml_results.txt", "w") as f:
    f.write("Balanced Logistic Regression\n\n")
    f.write("Train Results:\n")
    f.write(train_report)
    f.write("\n\nValidation Results:\n")
    f.write(val_report)
    f.write("\n\nTest Results:\n")
    f.write(test_report)

# Save model and vectorizer
joblib.dump(model, r"D:\IIT-Gandhinagar_Project\final_models\balanced_logistic_model.pkl")
joblib.dump(vectorizer, r"D:\IIT-Gandhinagar_Project\final_models\tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")