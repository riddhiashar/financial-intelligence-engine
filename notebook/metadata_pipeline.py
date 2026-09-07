import os
import re
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

LABEL_MAP  = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

print(f"Loading model: {MODEL_NAME}")
embedding_model = SentenceTransformer(MODEL_NAME)
print(f"Model ready — {embedding_model.get_embedding_dimension()} dims")


def extract_tickers(text):
    if not isinstance(text, str):
        return []
    return list(set(re.findall(r"\$[A-Z]{1,6}\b", str(text))))


def build_metadata(df, label_map=None):
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
            "sentiment":     label_map.get(
                                 int(row.get("label", -1)), "Unknown"),
            "word_count":    int(row.get("word_count", 0)),
            "token_count":   int(row.get("token_count", 0))
        }
        metadata.append(entry)
    return metadata


def build_vector_payload(metadata_entry, embedding):
    return {
        "id":     metadata_entry["id"],
        "vector": embedding.tolist(),
        "payload": {
            "original_text": metadata_entry["original_text"],
            "cleaned_text":  metadata_entry["cleaned_text"],
            "ticker":        metadata_entry["ticker"],
            "tickers_list":  metadata_entry["tickers_list"],
            "label":         metadata_entry["label"],
            "sentiment":     metadata_entry["sentiment"],
            "word_count":    metadata_entry["word_count"],
            "token_count":   metadata_entry["token_count"]
        }
    }


def generate_embeddings(texts, batch_size=BATCH_SIZE):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = embedding_model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        all_embeddings.append(embs)
    return np.vstack(all_embeddings)


def save_metadata(metadata, path="metadata.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path} — {len(metadata)} entries")


def load_metadata(path="metadata.json"):
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded: {path} — {len(metadata)} entries")
    return metadata


def embed_and_search(raw_text, embeddings, metadata,
                     top_k=5, min_similarity=0.3,
                     sentiment_filter=None):
    query_vec = embedding_model.encode(
        raw_text, convert_to_numpy=True
    ).reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings)[0]

    if sentiment_filter:
        valid = [
            i for i, m in enumerate(metadata)
            if m["sentiment"] == sentiment_filter
        ]
    else:
        valid = list(range(len(metadata)))

    filtered = sorted(
        [(i, sims[i]) for i in valid],
        key=lambda x: x[1], reverse=True
    )

    results = []
    for idx, sim in filtered:
        if len(results) >= top_k:
            break
        if sim < min_similarity:
            break
        results.append({
            "rank":       len(results) + 1,
            "similarity": float(sim),
            "headline":   metadata[idx]["original_text"],
            "cleaned":    metadata[idx]["cleaned_text"],
            "sentiment":  metadata[idx]["sentiment"],
            "ticker":     metadata[idx]["ticker"],
            "id":         idx
        })
    return results


if __name__ == "__main__":
    print("Running metadata pipeline...")
    df = pd.read_csv("day4_verified.csv")
    df["label"] = df["label"].astype(int)

    print(f"Building metadata for {len(df)} headlines...")
    metadata = build_metadata(df)
    save_metadata(metadata)

    print(f"Generating embeddings...")
    texts = df["cleaned_text"].astype(str).tolist()
    embeddings = generate_embeddings(texts)
    np.save("headline_embeddings.npy", embeddings)
    print(f"Saved embeddings: {embeddings.shape}")

    print("Pipeline complete.")
