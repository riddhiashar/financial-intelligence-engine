"""
rag_pipeline.py — RAG Summarization Pipeline
Week 6: LLM Integration
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from vector_db import qdrant_search, hybrid_search

# read from environment variables — critical for Docker
COLLECTION_NAME = "financial_headlines"
OLLAMA_MODEL    = "llama3"
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "localhost")

print(f"RAG Pipeline connecting to Qdrant at: {QDRANT_HOST}:6333")
qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
print(f"RAG Pipeline Qdrant connected")

print(f"Loading {OLLAMA_MODEL} via Ollama at: {OLLAMA_HOST}:11434")
llm = OllamaLLM(
    model       = OLLAMA_MODEL,
    base_url    = f"http://{OLLAMA_HOST}:11434",
    temperature = 0.1,
    num_predict = 120,
    stop        = [
        "\n\n\n",
        "Note:",
        "Disclaimer:",
        "Please note",
        "In conclusion",
        "Overall,"
    ]
)
print(f"LLM ready")

RAG_TEMPLATE = """You are a financial market intelligence analyst.

Read the {num_headlines} financial headlines below and write exactly 2 sentences with NO line break between them.

Sentence 1: Describe the dominant market sentiment and overall trend.
Sentence 2: Mention the most significant specific event or stock ticker.

Rules:
- Write ONLY the 2 sentences back to back with a single space between them.
- Do NOT add a line break between the two sentences.
- No introduction, no prefix, no notes, no conclusion.
- Start directly with the first sentence.
- Use only facts from the headlines below.

HEADLINES:
{context}

BRIEFING:"""

prompt_template = PromptTemplate(
    input_variables=["context", "num_headlines"],
    template=RAG_TEMPLATE
)

rag_chain = prompt_template | llm


def clean_summary(text: str) -> str:
    """Remove common Llama3 prefixes and markdown artifacts."""
    prefixes = [
        "Here is the market intelligence briefing:",
        "Here is the briefing:",
        "MARKET INTELLIGENCE BRIEFING:",
        "BRIEFING:",
        "Market Intelligence Briefing:",
    ]
    for prefix in prefixes:
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix):].strip()
    text = text.replace("**Market Sentiment and Trend:**", "")
    text = text.replace("**Significant Event/Ticker:**", "")
    text = text.replace("**Most Significant Specific Event:**", "")
    return text.strip()


def generate_market_summary(query: str,
                             top_k: int = 5,
                             sentiment_filter: str = None) -> dict:
    """
    Complete RAG pipeline: retrieve + augment + generate.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    start_time = time.time()

    results = hybrid_search(
        client           = qdrant_client,
        query            = query,
        top_k            = top_k,
        min_similarity   = 0.3,
        sentiment_filter = sentiment_filter,
        collection_name  = COLLECTION_NAME
    )

    if not results:
        return {
            "query":   query,
            "summary": "No relevant headlines found for this query.",
            "sources": [],
            "processing_time_ms": round(
                (time.time() - start_time) * 1000, 2
            )
        }

    retrieval_time = (time.time() - start_time) * 1000

    context = "\n".join([
        f"{i+1}. [{r['sentiment']}] {r['headline']} "
        f"(similarity: {r['similarity']:.2f})"
        for i, r in enumerate(results)
    ])

    llm_start = time.time()
    raw_summary = rag_chain.invoke({
        "context":       context,
        "num_headlines": len(results)
    })
    summary   = clean_summary(raw_summary.strip())
    llm_time  = (time.time() - llm_start) * 1000
    total_time = (time.time() - start_time) * 1000

    return {
        "query":             query,
        "sentiment_filter":  sentiment_filter,
        "summary":           summary,
        "sources":           [
            {
                "rank":           r["rank"],
                "similarity":     r["similarity"],
                "headline":       r["headline"],
                "sentiment":      r["sentiment"],
                "ticker":         r["ticker"],
                "retrieval_type": r.get("retrieval_type", "semantic")
            }
            for r in results
        ],
        "retrieval_time_ms":  round(retrieval_time, 2),
        "llm_time_ms":        round(llm_time, 2),
        "processing_time_ms": round(total_time, 2)
    }


if __name__ == "__main__":
    result = generate_market_summary("Tesla earnings revenue")
    print(f"Query: {result['query']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nTotal time: {result['processing_time_ms']}ms")