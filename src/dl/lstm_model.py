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

# Reproducibility

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Load Data

df = pd.read_csv(r"D:\IIT-Gandhinagar_Project\sample_100k.csv")

X = df["DATA"].astype(str)
y = df["TOPIC"]

# Encode labels
labels = {label: idx for idx, label in enumerate(y.unique())}
y_encoded = y.map(labels)


# Split (70 / 10 / 20)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=SEED, stratify=y_temp
)


# Tokenization

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

counter = Counter()
for text in X_train:
    counter.update(tokenize(text))

# limit vocab
MAX_VOCAB = 20000
vocab = {word: i+1 for i, (word, _) in enumerate(counter.most_common(MAX_VOCAB))}


# Encode + Pad

MAX_LEN = 100

def encode(text):
    return [vocab.get(word, 0) for word in tokenize(text)]

def pad(seqs):
    padded = []
    for seq in seqs:
        seq = seq[:MAX_LEN]
        seq = seq + [0]*(MAX_LEN - len(seq))
        padded.append(seq)
    return torch.tensor(padded)

X_train_seq = pad([encode(t) for t in X_train])
X_val_seq = pad([encode(t) for t in X_val])
X_test_seq = pad([encode(t) for t in X_test])

y_train = torch.tensor(y_train.values)
y_val = torch.tensor(y_val.values)
y_test = torch.tensor(y_test.values)

# Model

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

model = LSTMModel(
    vocab_size=MAX_VOCAB + 1,
    embed_dim=128,
    hidden_dim=128,
    num_classes=len(labels)
)


# Training

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 3
BATCH_SIZE = 256

def train_epoch(X, y):
    model.train()
    total_loss = 0
    for i in range(0, len(X), BATCH_SIZE):
        xb = X[i:i+BATCH_SIZE]
        yb = y[i:i+BATCH_SIZE]

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss

for epoch in range(EPOCHS):
    loss = train_epoch(X_train_seq, y_train)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Evaluation

def evaluate(X, y):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i+BATCH_SIZE]
            outputs = model(xb)
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.numpy())
    return classification_report(y.numpy(), preds)

train_report = evaluate(X_train_seq, y_train)
val_report = evaluate(X_val_seq, y_val)
test_report = evaluate(X_test_seq, y_test)

print("\nTRAIN RESULTS:\n", train_report)
print("\nVALIDATION RESULTS:\n", val_report)
print("\nTEST RESULTS:\n", test_report)

# Save Results

with open(r"D:\IIT-Gandhinagar_Project\experiments\dl_results.txt", "a") as f:
    f.write("\n" + "="*80 + "\n")
    f.write("LSTM RESULTS\n\n")
    f.write("TRAIN:\n" + train_report + "\n\n")
    f.write("VALIDATION:\n" + val_report + "\n\n")
    f.write("TEST:\n" + test_report + "\n")


# Save Model + Vocab

torch.save(model.state_dict(), r"D:\IIT-Gandhinagar_Project\final_models\lstm_model.pth")
joblib.dump(vocab, r"D:\IIT-Gandhinagar_Project\final_models\vocab.pkl")
joblib.dump(labels, r"D:\IIT-Gandhinagar_Project\final_models\label_map.pkl")

print("\nLSTM model, vocab, and labels saved successfully.")