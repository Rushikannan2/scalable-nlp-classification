import argparse
import pickle
import torch
import re
import sys
import os

# -------------------------
# FIX IMPORT PATH
# -------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------------
# GLOBAL CACHE (IMPORTANT)
# -------------------------
CACHE = {}

# -------------------------
# loader
# -------------------------
def load_any(path):
    import joblib
    try:
        return pickle.load(open(path, "rb"))
    except:
        return joblib.load(path)

# -------------------------
# preprocess
# -------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# -------------------------
# encode
# -------------------------
def encode(text, vocab, max_len):
    tokens = preprocess(text).split()
    ids = [vocab.get(t, 0) for t in tokens[:max_len]]
    return ids + [0] * (max_len - len(ids))

# -------------------------
# decode
# -------------------------
def decode(pred, inv):
    return inv[int(pred)]

# -------------------------
# ML
# -------------------------
def ml_predict(text, model, vec, inv):
    X = vec.transform([preprocess(text)])
    return decode(model.predict(X)[0], inv)

# -------------------------
# LOAD MODELS ONCE
# -------------------------
def get_lstm():
    if "lstm" not in CACHE:
        from dl.lstm_model import LSTMModel
        device = "cuda" if torch.cuda.is_available() else "cpu"

        vocab = load_any("final_models/lstm_vocab.pkl")
        inv = load_any("final_models/lstm_inverse_labels.pkl")

        model = LSTMModel(len(vocab)+1, 128, 128, len(inv))
        model.load_state_dict(torch.load("final_models/lstm_final.pth", map_location=device))
        model.to(device).eval()

        CACHE["lstm"] = (model, vocab, inv, device)
    return CACHE["lstm"]

def get_bilstm():
    if "bilstm" not in CACHE:
        from dl.bi_lstm_model import BiLSTMModel
        device = "cuda" if torch.cuda.is_available() else "cpu"

        vocab = load_any("final_models/lstm_vocab.pkl")
        inv = load_any("final_models/lstm_inverse_labels.pkl")

        model = BiLSTMModel(30001, 200, 256, len(inv), 2)
        model.load_state_dict(torch.load("final_models/bilstm_best.pth", map_location=device))
        model.to(device).eval()

        CACHE["bilstm"] = (model, vocab, inv, device)
    return CACHE["bilstm"]

def get_fasttext():
    if "fasttext" not in CACHE:
        from ml.Fasttext import FastText
        device = "cuda" if torch.cuda.is_available() else "cpu"

        vocab = load_any("final_models/fasttext_vocab.pkl")
        inv = load_any("final_models/fasttext_inverse_labels.pkl")

        model = FastText(len(vocab), 128, len(inv))
        model.load_state_dict(torch.load("final_models/fasttext_model.pt", map_location=device))
        model.to(device).eval()

        CACHE["fasttext"] = (model, vocab, inv, device)
    return CACHE["fasttext"]

# -------------------------
# FEATURE
# -------------------------
def features(text):
    w = text.split()
    return [len(w), sum(len(x) for x in w)/len(w) if w else 0]

# -------------------------
# PREDICT
# -------------------------
def predict(text, model):

    if model == "svm":
        return ml_predict(text,
            load_any("final_models/svm_model.pkl"),
            load_any("final_models/tfidf_vectorizer.pkl"),
            load_any("final_models/inverse_label_map.pkl"))

    elif model == "logistic":
        return ml_predict(text,
            load_any("final_models/normal_logistic_model.pkl"),
            load_any("final_models/tfidf_vectorizer_normal.pkl"),
            load_any("final_models/inverse_label_map_normal.pkl"))

    elif model == "balanced_logistic":
        return ml_predict(text,
            load_any("final_models/balanced_logistic_model.pkl"),
            load_any("final_models/tfidf_vectorizer_balanced.pkl"),
            load_any("final_models/inverse_label_map_balanced.pkl"))

    elif model == "nb":
        return ml_predict(text,
            load_any("final_models/nb_best_model.pkl"),
            load_any("final_models/nb_best_vectorizer.pkl"),
            load_any("final_models/nb_inverse_labels.pkl"))

    elif model == "lstm":
        model, vocab, inv, device = get_lstm()
        seq = torch.tensor([encode(text, vocab, 100)]).to(device)
        return decode(torch.argmax(model(seq), 1).item(), inv)

    elif model == "bilstm":
        model, vocab, inv, device = get_bilstm()
        seq = torch.tensor([encode(text, vocab, 120)]).to(device)
        feat = torch.tensor([features(preprocess(text))], dtype=torch.float).to(device)
        return decode(torch.argmax(model(seq, feat), 1).item(), inv)

    elif model == "fasttext":
        model, vocab, inv, device = get_fasttext()
        seq = torch.tensor([encode(text, vocab, 50)]).to(device)
        return decode(torch.argmax(model(seq), 1).item(), inv)

# -------------------------
# RUN ALL
# -------------------------
def run_all(text):
    models = ["svm","logistic","balanced_logistic","nb","fasttext","lstm","bilstm"]
    print("\nInput:", text)
    print("\nPredictions:\n")
    for m in models:
        print(f"{m:20s} -> {predict(text, m)}")

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str)
    parser.add_argument("--model_type", type=str, default="all")

    args = parser.parse_args()
    text = args.text if args.text else "government policy impacts economy"

    if args.model_type == "all":
        run_all(text)
    else:
        print(predict(text, args.model_type))