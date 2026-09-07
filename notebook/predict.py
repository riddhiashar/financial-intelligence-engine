import os
import re
import torch
import numpy as np
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

LABEL_MAP  = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
NUM_LABELS = 3
MAX_LENGTH = 23
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "distilbert_tokenizer")
MODEL_PATH     = os.path.join(BASE_DIR, "distilbert_best_model.pt")

print(f"Loading tokenizer from: {TOKENIZER_PATH}")
tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)

print(f"Loading model from: {MODEL_PATH}")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=NUM_LABELS
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"Model ready on: {DEVICE}")


def remove_urls(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    return text.strip()

def remove_twitter_artifacts(text):
    text = re.sub(r"^RT @\w+:\s*", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("&amp;", "and")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = re.sub(r"#(\w+)", r"\1", text)
    return text.strip()

def fix_encoding(text):
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

def normalize_tickers(text):
    def uppercase_ticker(match):
        return match.group(0).upper()
    text = re.sub(r"\$[a-zA-Z]{1,6}\b", uppercase_ticker, text)
    return text

def lowercase_except_tickers(text):
    tickers = re.findall(r"\$[A-Z]{1,6}\b", text)
    placeholders = {}
    for i, ticker in enumerate(tickers):
        placeholder = f"__ticker{i}__"
        placeholders[placeholder] = ticker
        text = text.replace(ticker, placeholder, 1)
    text = text.lower()
    for placeholder, ticker in placeholders.items():
        text = text.replace(placeholder, ticker)
    return text

def remove_special_chars(text):
    text = re.sub(r"[^a-zA-Z0-9\s\$\%\.\,\-\']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def full_clean_pipeline(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\n\t\r]", " ", text)
    text = remove_urls(text)
    text = remove_twitter_artifacts(text)
    text = fix_encoding(text)
    text = normalize_tickers(text)
    text = lowercase_except_tickers(text)
    text = remove_special_chars(text)
    return text.strip()


def predict(text: str) -> dict:
    if not text or not isinstance(text, str):
        return {
            "error": "Invalid input",
            "original_text": str(text)
        }
    cleaned = full_clean_pipeline(text)
    if not cleaned:
        return {
            "error": "Text became empty after cleaning",
            "original_text": text,
            "cleaned_text": cleaned
        }
    encoding = tokenizer(
        cleaned,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits
    probabilities   = torch.softmax(logits, dim=1)[0].cpu().numpy()
    predicted_label = int(np.argmax(probabilities))
    confidence      = float(probabilities[predicted_label])
    return {
        "original_text": text,
        "cleaned_text":  cleaned,
        "label":         predicted_label,
        "sentiment":     LABEL_MAP[predicted_label],
        "confidence":    f"{confidence*100:.1f}%",
        "scores": {
            LABEL_MAP[i]: f"{probabilities[i]*100:.1f}%"
            for i in range(NUM_LABELS)
        }
    }


def batch_predict(texts: list) -> list:
    if not texts or not isinstance(texts, list):
        return [{"error": "Input must be a non-empty list"}]
    cleaned_texts = [full_clean_pipeline(t) for t in texts]
    valid_indices = [i for i, t in enumerate(cleaned_texts) if t]
    valid_texts   = [cleaned_texts[i] for i in valid_indices]
    if not valid_texts:
        return [{"error": "All texts became empty after cleaning"}]
    encodings = tokenizer(
        valid_texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encodings["input_ids"].to(DEVICE)
    attention_mask = encodings["attention_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    results   = []
    valid_idx = 0
    for i, original_text in enumerate(texts):
        if i not in valid_indices:
            results.append({
                "original_text": original_text,
                "error": "Text became empty after cleaning"
            })
        else:
            probs           = probabilities[valid_idx]
            predicted_label = int(np.argmax(probs))
            confidence      = float(probs[predicted_label])
            results.append({
                "original_text": original_text,
                "cleaned_text":  cleaned_texts[i],
                "label":         predicted_label,
                "sentiment":     LABEL_MAP[predicted_label],
                "confidence":    f"{confidence*100:.1f}%",
                "scores": {
                    LABEL_MAP[j]: f"{probs[j]*100:.1f}%"
                    for j in range(NUM_LABELS)
                }
            })
            valid_idx += 1
    return results


if __name__ == "__main__":
    test = "$AAPL beats earnings estimates revenue up 12%"
    print(f"Test input: {test}")
    print(f"Result:     {predict(test)}")
