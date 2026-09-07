"""
test_rag.py — Automated Tests for RAG Pipeline
Week 6 Day 4

Run with: pytest test_rag.py -v
Make sure API is running first:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Note: LLM tests take 20-40 seconds each on CPU.
"""

import pytest
import requests
import time

BASE_URL = "http://localhost:8000"
TIMEOUT  = 120  # 120 seconds for LLM calls


# ─── Health + RAG Availability Tests ─────────────────────

def test_health_llm_loaded():
    """Verify API is running and LLM is loaded."""
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["qdrant"] == "connected"


# ─── POST /summarize — Valid Input Tests ──────────────────

def test_summarize_returns_200():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "Tesla earnings revenue", "top_k": 3},
        timeout=TIMEOUT
    )
    assert response.status_code == 200


def test_summarize_returns_summary():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "Tesla earnings revenue", "top_k": 3},
        timeout=TIMEOUT
    )
    data = response.json()
    assert "summary" in data
    assert len(data["summary"]) > 0


def test_summarize_summary_not_empty():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "Apple iPhone revenue", "top_k": 3},
        timeout=TIMEOUT
    )
    data = response.json()
    assert data["summary"] != ""
    assert data["summary"] != "No relevant headlines found for this query."


def test_summarize_returns_sources():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "stock market crash", "top_k": 5},
        timeout=TIMEOUT
    )
    data = response.json()
    assert "sources" in data
    assert len(data["sources"]) > 0


def test_summarize_sources_have_correct_keys():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "Tesla earnings", "top_k": 3},
        timeout=TIMEOUT
    )
    data = response.json()
    if data["sources"]:
        source = data["sources"][0]
        assert "rank"       in source
        assert "similarity" in source
        assert "headline"   in source
        assert "sentiment"  in source
        assert "ticker"     in source


def test_summarize_respects_top_k():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "earnings revenue", "top_k": 3},
        timeout=TIMEOUT
    )
    data = response.json()
    assert len(data["sources"]) <= 3


def test_summarize_returns_request_id():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "Tesla earnings", "top_k": 3},
        timeout=TIMEOUT
    )
    data = response.json()
    assert "request_id" in data
    assert len(data["request_id"]) > 0


def test_summarize_returns_timing():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "Tesla earnings", "top_k": 3},
        timeout=TIMEOUT
    )
    data = response.json()
    assert "retrieval_time_ms"  in data
    assert "llm_time_ms"        in data
    assert "processing_time_ms" in data
    assert data["retrieval_time_ms"]  > 0
    assert data["llm_time_ms"]        > 0
    assert data["processing_time_ms"] > 0


def test_summarize_returns_timestamp():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "Tesla earnings", "top_k": 3},
        timeout=TIMEOUT
    )
    data = response.json()
    assert "timestamp" in data


def test_summarize_bearish_filter():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={
            "query":            "recession market crash",
            "top_k":            5,
            "sentiment_filter": "Bearish"
        },
        timeout=TIMEOUT
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_filter"] == "Bearish"
    for source in data["sources"]:
        assert source["sentiment"] == "Bearish"


def test_summarize_bullish_filter():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={
            "query":            "company beats earnings",
            "top_k":            5,
            "sentiment_filter": "Bullish"
        },
        timeout=TIMEOUT
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_filter"] == "Bullish"
    for source in data["sources"]:
        assert source["sentiment"] == "Bullish"


def test_summarize_neutral_filter():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={
            "query":            "Federal Reserve interest rate",
            "top_k":            5,
            "sentiment_filter": "Neutral"
        },
        timeout=TIMEOUT
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_filter"] == "Neutral"


def test_summarize_sources_sorted_by_similarity():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "Tesla stock earnings", "top_k": 5},
        timeout=TIMEOUT
    )
    data = response.json()
    if len(data["sources"]) > 1:
        sims = [s["similarity"] for s in data["sources"]]
        assert all(
            sims[i] >= sims[i+1]
            for i in range(len(sims)-1)
        )


# ─── POST /summarize — Invalid Input Tests ────────────────

def test_summarize_empty_query_returns_422():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": ""},
        timeout=10
    )
    assert response.status_code == 422


def test_summarize_whitespace_query_returns_422():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "   "},
        timeout=10
    )
    assert response.status_code == 422


def test_summarize_invalid_sentiment_returns_422():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={
            "query":            "Tesla earnings",
            "sentiment_filter": "Invalid"
        },
        timeout=10
    )
    assert response.status_code == 422


def test_summarize_top_k_too_large_returns_422():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "Tesla earnings", "top_k": 100},
        timeout=10
    )
    assert response.status_code == 422


def test_summarize_top_k_zero_returns_422():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "Tesla earnings", "top_k": 0},
        timeout=10
    )
    assert response.status_code == 422


def test_summarize_missing_query_returns_422():
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"top_k": 5},
        timeout=10
    )
    assert response.status_code == 422


def test_summarize_no_results_returns_message():
    """Very high similarity threshold should return no results."""
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={"query": "xyzabc123nonsensequery", "top_k": 5},
        timeout=TIMEOUT
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == [] or \
           "No relevant headlines" in data["summary"]