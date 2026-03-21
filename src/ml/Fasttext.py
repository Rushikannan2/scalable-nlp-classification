import pandas as pd
import numpy as np
import re
import os
import joblib
import random
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

# REPRODUCIBILITY

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


# TOKENIZATION

def tokenize(text):
    return text.split()


# VOCAB BUILDING

counter = Counter()

for text in X_train:
    counter.update(tokenize(text))

MAX_WORDS = 20000
vocab = {w: i+1 for i, (w, _) in enumerate(counter.most_common(MAX_WORDS))}
vocab["<PAD>"] = 0

VOCAB_SIZE = len(vocab)


# ENCODING

MAX_LEN = 50

def encode(text):
    tokens = tokenize(text)
    ids = [vocab.get(tok, 0) for tok in tokens[:MAX_LEN]]
    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    return ids

X_train_enc = np.array([encode(t) for t in X_train])
X_val_enc = np.array([encode(t) for t in X_val])
X_test_enc = np.array([encode(t) for t in X_test])

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)


# FASTTEXT MODEL (EFFICIENT)

class FastText(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        return self.fc(x)

EMBED_DIM = 128
NUM_CLASSES = len(labels)

model = FastText(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES)


# PARAMETER COUNT

total_params = sum(p.numel() for p in model.parameters())

print("\n===== MODEL PARAMETERS =====")
print(f"Vocab Size: {VOCAB_SIZE}")
print(f"Embedding Dim: {EMBED_DIM}")
print(f"Classes: {NUM_CLASSES}")
print(f"Total Parameters: {total_params:,}")


# TRAINING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_epoch(X, y):
    model.train()
    total_loss = 0

    for i in range(0, len(X), 256):
        xb = torch.tensor(X[i:i+256]).to(device)
        yb = torch.tensor(y[i:i+256]).to(device)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

def eval_loss(X, y):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.tensor(X[i:i+256]).to(device)
            yb = torch.tensor(y[i:i+256]).to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item()
    return total_loss

best_val = float('inf')

for epoch in range(10):
    train_loss = train_epoch(X_train_enc, y_train)
    val_loss = eval_loss(X_val_enc, y_val)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, "fasttext_best.pt"))
        print("Best model saved")

# EVALUATION

def predict(X):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.tensor(X[i:i+256]).to(device)
            out = model(xb)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
    return np.array(preds)

def evaluate(name, y_true, y_pred):
    print(f"\n========== {name} RESULTS ==========\n")

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

    return f"Accuracy: {acc:.4f}\n\n{report}"

# Load best model
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "fasttext_best.pt")))

train_pred = predict(X_train_enc)
val_pred = predict(X_val_enc)
test_pred = predict(X_test_enc)

train_report = evaluate("TRAIN", y_train, train_pred)
val_report = evaluate("VALIDATION", y_val, val_pred)
test_report = evaluate("TEST", y_test, test_pred)


# SAVE RESULTS

with open(RESULT_PATH, "a") as f:
    f.write("\n" + "="*80 + "\n")
    f.write("FASTTEXT (FINAL)\n")
    f.write("="*80 + "\n\n")

    f.write(f"Vocab Size: {VOCAB_SIZE}\n")
    f.write(f"Embedding Dim: {EMBED_DIM}\n")
    f.write(f"Classes: {NUM_CLASSES}\n")
    f.write(f"Total Parameters: {total_params}\n\n")

    f.write("TRAIN RESULTS:\n" + train_report + "\n\n")
    f.write("VALIDATION RESULTS:\n" + val_report + "\n\n")
    f.write("TEST RESULTS:\n" + test_report + "\n")

# SAVE EVERYTHING

torch.save(model.state_dict(), os.path.join(MODEL_PATH, "fasttext_model.pt"))

joblib.dump(vocab, os.path.join(MODEL_PATH, "fasttext_vocab.pkl"))
joblib.dump(labels, os.path.join(MODEL_PATH, "fasttext_label_map.pkl"))

inv_labels = {v: k for k, v in labels.items()}
joblib.dump(inv_labels, os.path.join(MODEL_PATH, "fasttext_inverse_labels.pkl"))

joblib.dump({
    "model": "FastText",
    "embedding_dim": EMBED_DIM,
    "max_len": MAX_LEN,
    "vocab_size": VOCAB_SIZE,
    "seed": SEED
}, os.path.join(MODEL_PATH, "fasttext_config.pkl"))

joblib.dump({
    "steps": [
        "lowercase",
        "remove_urls",
        "remove_numbers",
        "remove_special_characters",
        "normalize_spaces"
    ]
}, os.path.join(MODEL_PATH, "fasttext_preprocessing.pkl"))

joblib.dump(y.value_counts().to_dict(),
            os.path.join(MODEL_PATH, "fasttext_label_distribution.pkl"))

print("\n FASTTEXT: EVERYTHING SAVED SUCCESSFULLY")