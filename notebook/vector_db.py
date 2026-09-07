"""
vector_db.py — Vector Database Utility
Week 4 + Week 7: Ticker-aware retrieval + Hybrid search
"""

import os
import re
import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "financial_headlines"
VECTOR_SIZE     = 384
BATCH_SIZE      = 100
MODEL_NAME      = "all-MiniLM-L6-v2"

print(f"Loading embedding model: {MODEL_NAME}")
embedding_model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded — {embedding_model.get_embedding_dimension()} dims")


def get_client(host="localhost", port=6333):
    """Connect to Qdrant instance."""
    client = QdrantClient(host=host, port=port)
    print(f"Connected to Qdrant at {host}:{port}")
    return client


def extract_tickers_from_query(query: str) -> list:
    if not isinstance(query, str):
        return []
    return list(set(re.findall(r"\$[A-Z]{1,6}\b", query)))


def create_collection(client, collection_name=COLLECTION_NAME,
                      vector_size=VECTOR_SIZE):
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    print(f"Collection created: {collection_name}")


def upload_embeddings(client, embeddings, metadata,
                      collection_name=COLLECTION_NAME,
                      batch_size=BATCH_SIZE):
    total    = len(metadata)
    uploaded = 0
    for i in range(0, total, batch_size):
        batch_emb  = embeddings[i:i + batch_size]
        batch_meta = metadata[i:i + batch_size]
        points = [
            PointStruct(
                id=m["id"],
                vector=e.tolist(),
                payload={
                    "original_text": m["original_text"],
                    "cleaned_text":  m["cleaned_text"],
                    "ticker":        m["ticker"],
                    "tickers_list":  m["tickers_list"],
                    "label":         m["label"],
                    "sentiment":     m["sentiment"],
                    "word_count":    m["word_count"],
                    "token_count":   m["token_count"]
                }
            )
            for e, m in zip(batch_emb, batch_meta)
        ]
        client.upsert(collection_name=collection_name, points=points)
        uploaded += len(points)
    print(f"Uploaded {uploaded} points to {collection_name}")
    return uploaded


def qdrant_search(client, query, top_k=5,
                  min_similarity=0.3,
                  sentiment_filter=None,
                  ticker_filter=None,
                  collection_name=COLLECTION_NAME):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    query_vector = embedding_model.encode(
        query, convert_to_numpy=True
    ).tolist()

    must_conditions = []
    if sentiment_filter:
        must_conditions.append(
            FieldCondition(
                key="sentiment",
                match=MatchValue(value=sentiment_filter)
            )
        )
    if ticker_filter:
        must_conditions.append(
            FieldCondition(
                key="ticker",
                match=MatchValue(value=ticker_filter)
            )
        )

    query_filter = Filter(must=must_conditions) \
        if must_conditions else None

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
        score_threshold=min_similarity
    )

    return [
        {
            "rank":       rank + 1,
            "similarity": float(p.score),
            "headline":   p.payload["original_text"],
            "cleaned":    p.payload["cleaned_text"],
            "sentiment":  p.payload["sentiment"],
            "ticker":     p.payload["ticker"],
            "id":         p.id
        }
        for rank, p in enumerate(response.points)
    ]


def ticker_aware_search(client, query, top_k=5,
                         min_similarity=0.3,
                         sentiment_filter=None,
                         collection_name=COLLECTION_NAME):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    tickers  = extract_tickers_from_query(query)
    results  = []
    seen_ids = set()

    if tickers:
        for ticker in tickers:
            ticker_results = qdrant_search(
                client=client,
                query=query,
                top_k=top_k,
                min_similarity=min_similarity,
                sentiment_filter=sentiment_filter,
                ticker_filter=ticker,
                collection_name=collection_name
            )
            for r in ticker_results:
                if r["id"] not in seen_ids:
                    r["retrieval_type"] = "ticker_match"
                    results.append(r)
                    seen_ids.add(r["id"])

    remaining = top_k - len(results)
    if remaining > 0:
        general_results = qdrant_search(
            client=client,
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            sentiment_filter=sentiment_filter,
            ticker_filter=None,
            collection_name=collection_name
        )
        for r in general_results:
            if r["id"] not in seen_ids:
                r["retrieval_type"] = "semantic"
                results.append(r)
                seen_ids.add(r["id"])
            if len(results) >= top_k:
                break

    results.sort(key=lambda x: x["similarity"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results[:top_k]


def hybrid_search(client, query, top_k=5,
                  min_similarity=0.3,
                  sentiment_filter=None,
                  collection_name=COLLECTION_NAME):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    semantic_results = ticker_aware_search(
        client=client,
        query=query,
        top_k=top_k * 2,
        min_similarity=min_similarity * 0.8,
        sentiment_filter=sentiment_filter,
        collection_name=collection_name
    )

    query_keywords = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
    )
    stopwords = {
        "the", "and", "for", "are", "was", "were",
        "has", "had", "have", "that", "this", "with",
        "from", "not", "but", "its", "after"
    }
    query_keywords -= stopwords

    boosted_results = []
    for r in semantic_results:
        headline_lower = r["cleaned"].lower()
        boost = 0.0
        keyword_hits = sum(
            1 for kw in query_keywords
            if kw in headline_lower
        )
        if query_keywords:
            boost = min(
                0.1,
                keyword_hits / len(query_keywords) * 0.1
            )
        tickers = extract_tickers_from_query(query)
        for ticker in tickers:
            if ticker in r["headline"]:
                boost += 0.05

        r["hybrid_score"]  = round(r["similarity"] + boost, 4)
        r["keyword_boost"] = round(boost, 4)
        r["keyword_hits"]  = keyword_hits
        boosted_results.append(r)

    boosted_results.sort(
        key=lambda x: x["hybrid_score"], reverse=True
    )
    final = boosted_results[:top_k]
    for i, r in enumerate(final):
        r["rank"] = i + 1

    return final


if __name__ == "__main__":
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    client = get_client(host=QDRANT_HOST)

    print("\n--- Standard Search ---")
    results = qdrant_search(client, "Tesla earnings revenue", top_k=3)
    for r in results:
        print(f"  {r['rank']}. [{r['sentiment']}] "
              f"(sim={r['similarity']:.3f}) "
              f"{r['headline'][:60]}")