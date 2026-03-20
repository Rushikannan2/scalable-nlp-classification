import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import re
import os


# Reproducibility

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths

BASE_PATH = r"D:\IIT-Gandhinagar_Project"
MODEL_PATH = os.path.join(BASE_PATH, "final_models")
RESULT_PATH = os.path.join(BASE_PATH, "experiments", "dl_results.txt")

os.makedirs(MODEL_PATH, exist_ok=True)

# Load Data

df = pd.read_csv(os.path.join(BASE_PATH, "sample_100k.csv"))

X = df["DATA"].astype(str)
y = df["TOPIC"]


# Label Encoding

labels = {label: idx for idx, label in enumerate(y.unique())}
y_encoded = y.map(labels)

# Split (70 / 10 / 20)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=SEED, stratify=y_temp
)


# Tokenization (unigram + bigram)

def tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    bigrams = [words[i] + "_" + words[i+1] for i in range(len(words)-1)]
    return words + bigrams

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
    padded = []
    for seq in seqs:
        seq = seq[:MAX_LEN]
        seq = seq + [0]*(MAX_LEN - len(seq))
        padded.append(seq)
    return torch.tensor(padded, dtype=torch.long)

X_train_seq = pad([encode(t) for t in X_train]).to(device)
X_val_seq = pad([encode(t) for t in X_val]).to(device)
X_test_seq = pad([encode(t) for t in X_test]).to(device)

y_train = torch.tensor(y_train.values).to(device)
y_val = torch.tensor(y_val.values).to(device)
y_test = torch.tensor(y_test.values).to(device)


# BiLSTM Model

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        h = torch.cat((hidden[-2], hidden[-1]), dim=1)
        x = self.dropout(h)
        return self.fc(x)

model = BiLSTMModel(
    vocab_size=MAX_VOCAB + 1,
    embed_dim=200,
    hidden_dim=256,
    num_classes=len(labels)
).to(device)


# Training Setup

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

EPOCHS = 10
BATCH_SIZE = 256
PATIENCE = 2

best_val_loss = float('inf')
patience_counter = 0


# Training Functions

def train_epoch():
    model.train()
    total_loss = 0
    for i in range(0, len(X_train_seq), BATCH_SIZE):
        xb = X_train_seq[i:i+BATCH_SIZE]
        yb = y_train[i:i+BATCH_SIZE]

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss


def eval_loss(X, y):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i+BATCH_SIZE]
            yb = y[i:i+BATCH_SIZE]
            outputs = model(xb)
            loss = criterion(outputs, yb)
            total_loss += loss.item()
    return total_loss

# Training Loop (with early stopping)

for epoch in range(EPOCHS):
    train_loss = train_epoch()
    val_loss = eval_loss(X_val_seq, y_val)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save(model.state_dict(), os.path.join(MODEL_PATH, "bilstm_best.pth"))
        print("Best model saved")

    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping triggered")
        break

# Evaluation

def evaluate(X, y):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i+BATCH_SIZE]
            outputs = model(xb)
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())

    return classification_report(y.cpu().numpy(), preds, zero_division=0)

train_report = evaluate(X_train_seq, y_train)
val_report = evaluate(X_val_seq, y_val)
test_report = evaluate(X_test_seq, y_test)

print("\nTRAIN RESULTS:\n", train_report)
print("\nVALIDATION RESULTS:\n", val_report)
print("\nTEST RESULTS:\n", test_report)

# Save Results

with open(RESULT_PATH, "a") as f:
    f.write("\n" + "="*80 + "\n")
    f.write("FINAL BiLSTM (OPTIMIZED)\n\n")
    f.write("TRAIN:\n" + train_report + "\n\n")
    f.write("VALIDATION:\n" + val_report + "\n\n")
    f.write("TEST:\n" + test_report + "\n")

# Save Artifacts

joblib.dump(vocab, os.path.join(MODEL_PATH, "bilstm_vocab.pkl"))
joblib.dump(labels, os.path.join(MODEL_PATH, "bilstm_labels.pkl"))

print("\nAll artifacts saved successfully.")