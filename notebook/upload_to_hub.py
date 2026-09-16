"""
upload_to_hub.py — Upload fine-tuned DistilBERT to HuggingFace Hub

Run once:
    python upload_to_hub.py

Requires:
    pip install huggingface_hub
    HF_TOKEN environment variable or login via CLI
"""

import os
import torch
from huggingface_hub import HfApi, create_repo
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer
)

# ─── Configuration ────────────────────────────────────────
HF_USERNAME  = "your-huggingface-username"  # replace with yours
REPO_NAME    = "financial-sentiment-distilbert"
REPO_ID      = f"{HF_USERNAME}/{REPO_NAME}"

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(BASE_DIR, "distilbert_best_model.pt")
TOKENIZER_PATH    = os.path.join(BASE_DIR, "distilbert_tokenizer")
SAVE_DIR          = os.path.join(BASE_DIR, "hf_model_export")

LABEL2ID = {"Bearish": 0, "Bullish": 1, "Neutral": 2}
ID2LABEL = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

# ─── Step 1: Load your fine-tuned model ───────────────────
print("Loading fine-tuned DistilBERT model...")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels = 3,
    id2label   = ID2LABEL,
    label2id   = LABEL2ID
)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)
model.eval()
print("Model loaded successfully")

# ─── Step 2: Load tokenizer ───────────────────────────────
print("Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
print("Tokenizer loaded")

# ─── Step 3: Save in HuggingFace format ───────────────────
print(f"Saving model to {SAVE_DIR}...")
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Model saved in HuggingFace format")

# ─── Step 4: Create model card ────────────────────────────
model_card = """---
language: en
tags:
  - text-classification
  - financial
  - sentiment-analysis
  - distilbert
license: apache-2.0
datasets:
  - zeroshot/twitter-financial-news-sentiment
metrics:
  - accuracy
  - f1
---

# Financial Sentiment DistilBERT

Fine-tuned DistilBERT model for financial news sentiment classification.

## Model Description

This model classifies financial news headlines into three sentiment categories:
- **Bearish** (label 0) — Negative market sentiment
- **Bullish** (label 1) — Positive market sentiment  
- **Neutral** (label 2) — No clear directional sentiment

## Training Data

- Dataset: [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
- Training samples: 7,504 (after stratified split)
- Validation samples: 938
- Test samples: 938

## Performance

| Metric   | Score |
|----------|-------|
| Accuracy | 84.43% |
| Macro F1 | 0.80   |
| Bearish F1 | 0.74 |
| Bullish F1 | 0.77 |
| Neutral F1 | 0.90 |

## Usage

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="your-username/financial-sentiment-distilbert"
)

result = classifier("$AAPL beats earnings estimates revenue up 12%")
print(result)
# [{'label': 'Bullish', 'score': 0.99}]
```

## Training Details

- Base model: distilbert-base-uncased
- Max sequence length: 23 tokens
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 3
- Class imbalance handling: Class weights (Bearish 2.19x, Bullish 1.64x)

## Limitations

- Negation handling: Complex negations like "not expected to miss" may be misclassified
- Domain specific: Trained on Twitter financial news, may not generalize to formal reports
- Short text: Optimized for headlines (avg 11.7 tokens), not long documents
"""

with open(os.path.join(SAVE_DIR, "README.md"), "w") as f:
    f.write(model_card)
print("Model card created")

# ─── Step 5: Create repo and upload ──────────────────────
print(f"\nCreating HuggingFace repository: {REPO_ID}")
api = HfApi()

try:
    create_repo(
        repo_id  = REPO_ID,
        repo_type= "model",
        exist_ok = True,
        private  = False
    )
    print(f"Repository created: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"Repo may already exist: {e}")

print("Uploading model files...")
api.upload_folder(
    folder_path = SAVE_DIR,
    repo_id     = REPO_ID,
    repo_type   = "model"
)

print(f"\nModel uploaded successfully!")
print(f"View at: https://huggingface.co/{REPO_ID}")