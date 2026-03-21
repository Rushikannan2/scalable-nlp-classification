## Dataset Usage

The original dataset contains approximately **10 million samples (4GB)**.  
Due to computational constraints (GPU memory and training time), a **stratified subset of 100,000 samples** was used for all experiments.

This subset preserves the overall class distribution, enabling efficient experimentation while maintaining reliable performance evaluation.
---

## Repository Structure

```
IIT-Gandhinagar_Project/
├── src/
│   ├── data/
│   │   └── load_data.py              # Samples 100K rows from the .parquet file → .csv
│   ├── ml/
│   │   ├── svm_model.py              # TF-IDF + LinearSVC  (FINAL MODEL)
│   │   ├── Normal_Logistic_Regression.py
│   │   ├── Balanced_Logistic_Regression.py
│   │   ├── Bayes_classifier.py
│   │   └── Fasttext.py               # Custom FastText (no pretrained weights)
│   ├── dl/
│   │   ├── lstm_model.py             # Custom single-direction LSTM
│   │   └── bi_lstm_model.py          # Custom BiLSTM + handcrafted features
│   ├── Inference/
│   │   └── inference.py              # Unified inference for all 7 models
│   ├── utils/
│   │   ├── preprocessing.py
│   │   └── metrics.py
│   └── Visualation/
│       └── Analysis.ipynb            # Confusion matrices, class distribution plots
├── experiments/
│   ├── ml_results.txt                # Full classification reports for all ML models
│   └── dl_results.txt                # Full classification reports + loss curves for DL models
├── final_models/                     # Saved model artifacts
├── dataset_10M.parquet               # Full 10M corpus (gitignored)
├── sample_100k.csv                   # 100K stratified sample (gitignored)
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** All binary artifacts (`.csv`, `.parquet`) are excluded from the repository via `.gitignore`. You must generate `sample_100k.csv` and retrain to reproduce `final_models/`.

---

## Results Summary

| Model | Test Accuracy | Test Weighted F1 |
|---|---|---|
| **SVM — LinearSVC** *(final)* | **91.21%** | **0.91** |
| Logistic Regression | 90.11% | 0.90 |
| Balanced Logistic Regression | 88.00% | 0.89 |
| FastText (custom) | 87.01% | 0.86 |
| Naïve Bayes | 81.53% | 0.82 |
| LSTM (custom) | 80.20% | 0.83 |
| BiLSTM (custom) | 79.86% | 0.82 |

Dataset: 100K rows · 24 classes · Split: 70K train / 10K val / 20K test (stratified)

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or 3.11
- The raw dataset file `dataset_10M.parquet` placed in the project root
- (Optional) CUDA-capable GPU for DL model training

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:

| Package | Version |
|---|---|
| torch | 2.10.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.1 |
| numpy | 2.4.3 |
| pyarrow | 23.0.1 |
| joblib | 1.5.3 |

---

## Data Preparation

This step reads the raw `dataset_10M.parquet` file and writes a 100K-row CSV sample used by all training scripts.

```bash
python src/data/load_data.py
```

**Output:** `sample_100k.csv` in the project root.

> The script takes the first 100K rows using PyArrow batch streaming — no full-file load into RAM.

---

## Training Instructions

All scripts read from `sample_100k.csv` and save trained artifacts to `final_models/`.  
Results are appended to the corresponding file in `experiments/`.

Run each script from the **project root**:

### Classical ML Models

```bash
# SVM — LinearSVC  (FINAL MODEL)
python src/ml/svm_model.py

# Logistic Regression (standard)
python src/ml/Normal_Logistic_Regression.py

# Logistic Regression (class-balanced)
python src/ml/Balanced_Logistic_Regression.py

# Naïve Bayes
python src/ml/Bayes_classifier.py

# FastText (custom, no pretrained weights)
python src/ml/Fasttext.py
```

### Deep Learning Models

CUDA is used automatically if available; falls back to CPU.

```bash
# Single-direction LSTM
python src/dl/lstm_model.py

# Bidirectional LSTM + handcrafted features
python src/dl/bi_lstm_model.py
```

### Training Outputs

Each training script produces:

| Artifact | Location | Description |
|---|---|---|
| Trained model | `final_models/*.pkl` / `*.pt` / `*.pth` | Model weights |
| TF-IDF vectorizer | `final_models/tfidf_vectorizer.pkl` | Fitted on train split only |
| Label map | `final_models/label_map.pkl` | `{topic_string: int}` |
| Inverse label map | `final_models/inverse_label_map.pkl` | `{int: topic_string}` |
| Classification report | `experiments/ml_results.txt` or `dl_results.txt` | Per-class P/R/F1 + confusion matrix |

> All scripts fix `random_state=42` and set `numpy`, `torch`, and `random` seeds for full reproducibility.

---

## Inference Instructions

The unified inference script supports all 7 models via a single `--model_type` flag.

Run from the **project root**:

```bash
python src/Inference/inference.py --text "your input text here" --model_type svm
```

### Available `--model_type` values

| Value | Model |
|---|---|
| `svm` | TF-IDF + LinearSVC (recommended) |
| `logistic` | TF-IDF + Logistic Regression |
| `balanced_logistic` | TF-IDF + Balanced Logistic Regression |
| `nb` | TF-IDF + Naïve Bayes |
| `fasttext` | Custom FastText |
| `lstm` | Custom LSTM |
| `bilstm` | Custom BiLSTM |
| `all` | Run all 7 models and print each prediction |

### Examples

```bash
# Predict with the final SVM model
python src/Inference/inference.py --text "government policy impacts inflation" --model_type svm

# Predict with all models simultaneously
python src/Inference/inference.py --text "neural networks for image recognition" --model_type all

# Run with default text (uses built-in example if --text is omitted)
python src/Inference/inference.py --model_type svm
```

### Example Output

```
Input: government policy impacts inflation

Predictions:

svm                  -> Economics
```

When using `--model_type all`:

```
Input: government policy impacts inflation

Predictions:

svm                  -> Economics
logistic             -> Economics
balanced_logistic    -> Economics
nb                   -> Economics
fasttext             -> Economics
lstm                 -> Politics
bilstm               -> Economics
```

---

## Input / Output Schema

### Training Data (`sample_100k.csv`)

| Column | Type | Description |
|---|---|---|
| `DATA` | `str` | Raw text document |
| `TOPIC` | `str` | Topic label string (24 unique values) |

### Inference Input

| Parameter | Type | Description |
|---|---|---|
| `--text` | `str` | Any raw text string to classify |
| `--model_type` | `str` | One of: `svm`, `logistic`, `balanced_logistic`, `nb`, `fasttext`, `lstm`, `bilstm`, `all` |

### Inference Output

| Type | Description |
|---|---|
| `str` | Predicted topic label string (e.g. `"Economics"`, `"Technology"`) |

The preprocessing applied at inference is identical to training:
1. Lowercase
2. URL removal
3. Digit removal
4. Non-alphabetic character stripping
5. Whitespace normalisation

---

## Reproducibility

- All scripts set `random.seed(42)`, `numpy.random.seed(42)`, `torch.manual_seed(42)`, and `torch.cuda.manual_seed_all(42)`.
- `torch.backends.cudnn.deterministic = True` and `benchmark = False` are set in DL scripts.
- Data splits use `sklearn.model_selection.train_test_split` with `stratify=y` and `random_state=42` consistently across all scripts.
- The full pipeline runs end-to-end in order: **data prep → train → inference** with no manual intervention beyond activating the virtual environment.

---

### Report PDF: https://drive.google.com/file/d/1ETb9qB_lpLuG6O7oCTN63l7bNFkfujcL/view?usp=sharing


## Experimental Logs

Full per-class precision, recall, F1-score, and confusion matrices for every model and every split (train / validation / test) are saved in:

- `experiments/ml_results.txt` — SVM, Logistic Regression, Balanced LR, Naïve Bayes, FastText
- `experiments/dl_results.txt` — LSTM, BiLSTM (includes per-epoch loss curves)
