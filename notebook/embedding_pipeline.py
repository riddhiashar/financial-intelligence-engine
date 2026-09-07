import os
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer

LABEL_MAP  = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

print(f"Loading embedding model: {MODEL_NAME}")
embedding_model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded — dimensions: {embedding_model.get_embedding_dimension()}")


def extract_tickers(text):
    if not isinstance(text, str):
        return []
    return list(set(re.findall(r"\$[A-Z]{1,6}\b", str(text))))


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string."""
    if not text or not isinstance(text, str):
        return np.zeros(embedding_model.get_embedding_dimension())
    return embedding_model.encode(text, convert_to_numpy=True)


def embed_batch(texts: list) -> np.ndarray:
    """Embed a list of text strings efficiently."""
    if not texts:
        return np.array([])
    return embedding_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True
    )


def build_metadata(df, label_map=None):
    """Build metadata list from dataframe."""
    if label_map is None:
        label_map = LABEL_MAP
    metadata = []
    for idx, row in df.iterrows():
        tickers = extract_tickers(str(row.get("text", "")))
        entry = {
            "id":            int(idx),
            "original_text": str(row.get("text", "")),
            "cleaned_text":  str(row.get("cleaned_text", "")),
            "ticker":        ",".join(tickers),
            "tickers_list":  tickers,
            "label":         int(row.get("label", -1)),
            "sentiment":     label_map.get(int(row.get("label", -1)), "Unknown"),
            "word_count":    int(row.get("word_count", 0)),
            "token_count":   int(row.get("token_count", 0))
        }
        metadata.append(entry)
    return metadata


def save_embeddings(embeddings, path="headline_embeddings.npy"):
    """Save embeddings numpy array to disk."""
    np.save(path, embeddings)
    print(f"Saved embeddings: {path} — shape {embeddings.shape}")


def load_embeddings(path="headline_embeddings.npy"):
    """Load embeddings from disk."""
    embeddings = np.load(path)
    print(f"Loaded embeddings: {path} — shape {embeddings.shape}")
    return embeddings


def save_metadata(metadata, path="metadata.json"):
    """Save metadata list to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {path} — {len(metadata)} entries")


def load_metadata(path="metadata.json"):
    """Load metadata from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded metadata: {path} — {len(metadata)} entries")
    return metadata


if __name__ == "__main__":
    import pandas as pd
    print("Running embedding pipeline on day4_verified.csv...")
    df = pd.read_csv("day4_verified.csv")
    df["label"] = df["label"].astype(int)
    texts = df["cleaned_text"].astype(str).tolist()
    print(f"Generating embeddings for {len(texts)} headlines...")
    embeddings = embed_batch(texts)
    metadata = build_metadata(df)
    save_embeddings(embeddings)
    save_metadata(metadata)
    print("Pipeline complete.")
