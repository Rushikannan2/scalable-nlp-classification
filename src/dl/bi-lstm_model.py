import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import re
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter


# Reproducibility

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Paths

BASE_PATH = r"D:\IIT-Gandhinagar_Project"
DATA_PATH = os.path.join(BASE_PATH, "sample_100k.csv")
MODEL_PATH = os.path.join(BASE_PATH, "final_models")
RESULT_PATH = os.path.join(BASE_PATH, "experiments", "dl_results.txt")

os.makedirs(MODEL_PATH, exist_ok=True)

# Preprocessing

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Feature Engineering

def extract_features(text):
    words = text.split()
    return [
        len(words),
        np.mean([len(w) for w in words]) if words else 0,
        text.count('!'),
        text.count('?')
    ]


# Load Data

df = pd.read_csv(DATA_PATH)
df["DATA"] = df["DATA"].astype(str).apply(preprocess)

extra_features = np.array(df["DATA"].apply(extract_features).tolist())

X = df["DATA"]
y = df["TOPIC"]

# Label Encoding

labels = {label: idx for idx, label in enumerate(sorted(y.unique()))}
inv_labels = {v: k for k, v in labels.items()}
y_encoded = y.map(labels)


# Split

X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(
    X, y_encoded, extra_features,
    test_size=0.3, random_state=SEED, stratify=y_encoded
)

X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
    X_temp, y_temp, f_temp,
    test_size=2/3, random_state=SEED, stratify=y_temp
)

# Tokenization

def tokenize(text):
    words = text.split()
    bigrams = [words[i] + "_" + words[i+1] for i in range(len(words)-1)]
    return words + bigrams


# Vocabulary

counter = Counter()
for text in X_train:
    counter.update(tokenize(text))

MAX_VOCAB = 30000
vocab = {word: i+1 for i, (word, _) in enumerate(counter.most_common(MAX_VOCAB))}

# Encode + Pad

MAX_LEN = 120

def encode(text):
    return [vocab.get(word, 0) for word in tokenize(text)]

def pad(seqs):
    return torch.tensor([
        seq[:MAX_LEN] + [0]*(MAX_LEN - len(seq[:MAX_LEN]))
        for seq in seqs
    ], dtype=torch.long)

X_train_seq = pad([encode(t) for t in X_train])
X_val_seq   = pad([encode(t) for t in X_val])
X_test_seq  = pad([encode(t) for t in X_test])

f_train = torch.tensor(f_train, dtype=torch.float)
f_val   = torch.tensor(f_val, dtype=torch.float)
f_test  = torch.tensor(f_test, dtype=torch.float)

y_train = torch.tensor(y_train.values, dtype=torch.long)
y_val   = torch.tensor(y_val.values, dtype=torch.long)
y_test  = torch.tensor(y_test.values, dtype=torch.long)

# Class Weights

class_counts = np.bincount(y_train.numpy())
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Model

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, feat_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden_dim * 2 + feat_dim, 256)
        self.norm = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, features):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        h = torch.cat((hidden[-2], hidden[-1]), dim=1)
        x = torch.cat([h, features], dim=1)
        x = self.dropout(self.norm(self.fc1(x)))
        return self.fc2(x)

model = BiLSTMModel(
    vocab_size=MAX_VOCAB + 1,
    embed_dim=200,
    hidden_dim=256,
    num_classes=len(labels),
    feat_dim=4
).to(device)


# Training Setup

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

EPOCHS = 35
BATCH_SIZE = 256
PATIENCE = 5

best_val_loss = float('inf')
patience_counter = 0

history = {"train_loss": [], "val_loss": []}


# Training

def run_epoch(X, F, y, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    n_batches = 0

    with torch.set_grad_enabled(train):
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i+BATCH_SIZE].to(device)
            fb = F[i:i+BATCH_SIZE].to(device)
            yb = y[i:i+BATCH_SIZE].to(device)

            outputs = model(xb, fb)
            loss = criterion(outputs, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches

# Training Loop

for epoch in range(EPOCHS):
    train_loss = run_epoch(X_train_seq, f_train, y_train, True)
    val_loss   = run_epoch(X_val_seq, f_val, y_val, False)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, "bilstm_best.pth"))
        print(" Best model saved")
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(" Early stopping")
        break


# LOAD BEST MODEL

model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "bilstm_best.pth")))
print(" Loaded BEST model")


# Evaluation

def evaluate(X, F, y, name):
    model.eval()
    preds, probs = [], []

    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i+BATCH_SIZE].to(device)
            fb = F[i:i+BATCH_SIZE].to(device)

            outputs = model(xb, fb)
            prob = torch.softmax(outputs, dim=1)

            pred = torch.argmax(prob, dim=1)

            preds.extend(pred.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    report = classification_report(
        y.numpy(),
        preds,
        target_names=[str(inv_labels[i]) for i in range(len(inv_labels))],
        digits=2,
        zero_division=0
    )

    print(f"\n========== {name} RESULTS ==========\n")
    print(report)

    # Save outputs
    np.save(os.path.join(MODEL_PATH, f"{name}_preds.npy"), preds)
    np.save(os.path.join(MODEL_PATH, f"{name}_probs.npy"), probs)
    np.save(os.path.join(MODEL_PATH, f"{name}_labels.npy"), y.numpy())

    cm = confusion_matrix(y.numpy(), preds)
    np.save(os.path.join(MODEL_PATH, f"{name}_confusion.npy"), cm)

    return report

train_report = evaluate(X_train_seq, f_train, y_train, "TRAIN")
val_report   = evaluate(X_val_seq, f_val, y_val, "VALIDATION")
test_report  = evaluate(X_test_seq, f_test, y_test, "TEST")

# Save Everything

joblib.dump(vocab, os.path.join(MODEL_PATH, "vocab.pkl"))
joblib.dump(labels, os.path.join(MODEL_PATH, "labels.pkl"))
joblib.dump(inv_labels, os.path.join(MODEL_PATH, "inv_labels.pkl"))
joblib.dump(history, os.path.join(MODEL_PATH, "history.pkl"))

with open(RESULT_PATH, "a") as f:
    f.write("\n" + "="*80 + "\n")
    f.write("FINAL BiLSTM\n\n")
    f.write("TRAIN:\n" + train_report + "\n\n")
    f.write("VAL:\n" + val_report + "\n\n")
    f.write("TEST:\n" + test_report + "\n")

print("\n EVERYTHING SAVED PERFECTLY (FINAL VERSION)")