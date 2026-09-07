import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

print(f"Loading embedding model: {MODEL_NAME}")
embedding_model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded — {embedding_model.get_embedding_dimension()} dimensions")


def load_embeddings_and_metadata(
    embeddings_path="headline_embeddings.npy",
    metadata_path="metadata.json"
):
    embeddings = np.load(embeddings_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} embeddings and metadata entries")
    return embeddings, metadata


def semantic_search(query, embeddings, metadata,
                    top_k=5, min_similarity=0.3,
                    sentiment_filter=None):
    query_embedding = embedding_model.encode(
        query, convert_to_numpy=True
    ).reshape(1, -1)

    similarities = cosine_similarity(query_embedding, embeddings)[0]

    if sentiment_filter:
        valid_indices = [
            i for i, m in enumerate(metadata)
            if m["sentiment"] == sentiment_filter
        ]
    else:
        valid_indices = list(range(len(metadata)))

    filtered = [(i, similarities[i]) for i in valid_indices]
    filtered.sort(key=lambda x: x[1], reverse=True)

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
    embeddings, metadata = load_embeddings_and_metadata()
    queries = [
        "Tesla beats earnings expectations",
        "stock market crash recession fears",
        "Federal Reserve interest rate decision",
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        print("=" * 50)
        results = semantic_search(
            query, embeddings, metadata,
            top_k=3, min_similarity=0.3
        )
        for r in results:
            print(f"  {r['rank']}. [{r['sentiment']}] "
                  f"(sim={r['similarity']:.3f}) "
                  f"{r['headline'][:60]}")
