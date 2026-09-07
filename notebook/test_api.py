"""
test_api.py — Automated API Tests for Week 5
Run with: pytest test_api.py -v

Make sure API is running before running tests:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import pytest
import requests

BASE_URL = "http://localhost:8000"


def test_health_returns_200():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200

def test_health_returns_ok_status():
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    assert data["status"] == "ok"

def test_health_qdrant_connected():
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    assert data["qdrant"] == "connected"

def test_health_has_timestamp():
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    assert "timestamp" in data

def test_health_correct_collection():
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    assert data["collection"] == "financial_headlines"


# ─── POST /analyze Tests ──────────────────────────────────

def test_analyze_bullish_headline():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": "$AAPL beats earnings estimates revenue up 12%"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Bullish"

def test_analyze_bearish_headline():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": "$TSLA misses revenue target stock falls after hours"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Bearish"

def test_analyze_neutral_headline():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": "Federal Reserve announces interest rate decision next week"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Neutral"

def test_analyze_returns_confidence():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": "$MSFT price target raised by analysts strong buy"
    })
    assert response.status_code == 200
    data = response.json()
    assert "confidence" in data
    assert "%" in data["confidence"]

def test_analyze_returns_scores():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": "$AAPL beats earnings estimates"
    })
    assert response.status_code == 200
    data = response.json()
    assert "scores" in data
    assert "Bearish" in data["scores"]
    assert "Bullish" in data["scores"]
    assert "Neutral" in data["scores"]

def test_analyze_returns_similar_headlines():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": "$AAPL beats earnings estimates",
        "include_similar": True,
        "similar_limit": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert "similar_headlines" in data
    assert len(data["similar_headlines"]) <= 3

def test_analyze_similar_headlines_have_correct_keys():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": "$AAPL beats earnings estimates",
        "include_similar": True,
        "similar_limit": 3
    })
    data = response.json()
    if data["similar_headlines"]:
        headline = data["similar_headlines"][0]
        assert "rank" in headline
        assert "similarity" in headline
        assert "headline" in headline
        assert "sentiment" in headline
        assert "ticker" in headline

def test_analyze_returns_request_id():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": "$AAPL beats earnings"
    })
    data = response.json()
    assert "request_id" in data

def test_analyze_returns_processing_time():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": "$AAPL beats earnings"
    })
    data = response.json()
    assert "processing_time_ms" in data
    assert data["processing_time_ms"] > 0

def test_analyze_empty_text_returns_error():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": ""
    })
    assert response.status_code == 422

def test_analyze_whitespace_only_returns_error():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": "   "
    })
    assert response.status_code == 422

def test_analyze_missing_text_returns_error():
    response = requests.post(f"{BASE_URL}/analyze", json={})
    assert response.status_code == 422

def test_analyze_text_too_long_returns_error():
    long_text = "stock " * 200
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": long_text
    })
    assert response.status_code == 422

def test_analyze_without_similar():
    response = requests.post(f"{BASE_URL}/analyze", json={
        "text": "$AAPL beats earnings",
        "include_similar": False
    })
    assert response.status_code == 200
    data = response.json()
    assert data["similar_headlines"] == []


# ─── POST /batch Tests ────────────────────────────────────

def test_batch_returns_200():
    response = requests.post(f"{BASE_URL}/batch", json={
        "texts": [
            "$AAPL beats earnings estimates",
            "$TSLA misses revenue target",
            "Federal Reserve rate decision"
        ]
    })
    assert response.status_code == 200

def test_batch_returns_correct_count():
    texts = [
        "$AAPL beats earnings",
        "$TSLA misses revenue",
        "Fed rate decision"
    ]
    response = requests.post(f"{BASE_URL}/batch", json={
        "texts": texts
    })
    data = response.json()
    assert data["total"] == len(texts)

def test_batch_all_successful():
    response = requests.post(f"{BASE_URL}/batch", json={
        "texts": [
            "$AAPL beats earnings estimates",
            "$TSLA misses revenue target",
            "Federal Reserve rate decision"
        ]
    })
    data = response.json()
    assert data["successful"] == 3
    assert data["failed"] == 0

def test_batch_results_have_correct_sentiments():
    response = requests.post(f"{BASE_URL}/batch", json={
        "texts": [
            "$AAPL beats earnings estimates revenue up 12%",
            "$TSLA misses revenue target stock falls",
            "Federal Reserve announces rate decision"
        ]
    })
    data = response.json()
    results = data["results"]
    assert results[0]["sentiment"] == "Bullish"
    assert results[1]["sentiment"] == "Bearish"
    assert results[2]["sentiment"] == "Neutral"

def test_batch_empty_list_returns_error():
    response = requests.post(f"{BASE_URL}/batch", json={
        "texts": []
    })
    assert response.status_code == 422

def test_batch_too_many_items_returns_error():
    texts = ["stock market news"] * 51
    response = requests.post(f"{BASE_URL}/batch", json={
        "texts": texts
    })
    assert response.status_code == 422

def test_batch_returns_request_id():
    response = requests.post(f"{BASE_URL}/batch", json={
        "texts": ["$AAPL beats earnings"]
    })
    data = response.json()
    assert "request_id" in data

def test_batch_returns_processing_time():
    response = requests.post(f"{BASE_URL}/batch", json={
        "texts": ["$AAPL beats earnings"]
    })
    data = response.json()
    assert "processing_time_ms" in data
    assert data["processing_time_ms"] > 0


# ─── GET /search Tests ────────────────────────────────────

def test_search_returns_200():
    response = requests.get(
        f"{BASE_URL}/search",
        params={"query": "Tesla beats earnings"}
    )
    assert response.status_code == 200

def test_search_returns_list():
    response = requests.get(
        f"{BASE_URL}/search",
        params={"query": "Tesla beats earnings"}
    )
    data = response.json()
    assert isinstance(data, list)

def test_search_respects_limit():
    response = requests.get(
        f"{BASE_URL}/search",
        params={"query": "Tesla beats earnings", "limit": 3}
    )
    data = response.json()
    assert len(data) <= 3

def test_search_results_have_correct_keys():
    response = requests.get(
        f"{BASE_URL}/search",
        params={"query": "Tesla beats earnings"}
    )
    data = response.json()
    if data:
        result = data[0]
        assert "rank" in result
        assert "similarity" in result
        assert "headline" in result
        assert "sentiment" in result
        assert "ticker" in result
        assert "id" in result

def test_search_results_sorted_by_similarity():
    response = requests.get(
        f"{BASE_URL}/search",
        params={"query": "Tesla beats earnings", "limit": 5}
    )
    data = response.json()
    if len(data) > 1:
        sims = [r["similarity"] for r in data]
        assert all(
            sims[i] >= sims[i+1]
            for i in range(len(sims)-1)
        )

def test_search_sentiment_filter_bearish():
    response = requests.get(
        f"{BASE_URL}/search",
        params={
            "query":     "stock market crash",
            "sentiment": "Bearish",
            "limit":     5
        }
    )
    data = response.json()
    assert all(r["sentiment"] == "Bearish" for r in data)

def test_search_sentiment_filter_bullish():
    response = requests.get(
        f"{BASE_URL}/search",
        params={
            "query":     "company beats earnings",
            "sentiment": "Bullish",
            "limit":     5
        }
    )
    data = response.json()
    assert all(r["sentiment"] == "Bullish" for r in data)

def test_search_invalid_sentiment_returns_error():
    response = requests.get(
        f"{BASE_URL}/search",
        params={
            "query":     "Tesla earnings",
            "sentiment": "Invalid"
        }
    )
    assert response.status_code == 400

def test_search_invalid_ticker_format_returns_error():
    response = requests.get(
        f"{BASE_URL}/search",
        params={
            "query":  "Apple earnings",
            "ticker": "AAPL"
        }
    )
    assert response.status_code == 400

def test_search_valid_ticker_format():
    response = requests.get(
        f"{BASE_URL}/search",
        params={
            "query":  "Apple earnings",
            "ticker": "$AAPL"
        }
    )
    assert response.status_code == 200

def test_search_empty_query_returns_error():
    response = requests.get(
        f"{BASE_URL}/search",
        params={"query": ""}
    )
    assert response.status_code == 422

def test_search_high_similarity_threshold():
    response = requests.get(
        f"{BASE_URL}/search",
        params={
            "query":          "Tesla beats earnings",
            "min_similarity": 0.9
        }
    )
    data = response.json()
    assert all(r["similarity"] >= 0.9 for r in data)

def test_search_missing_query_returns_error():
    response = requests.get(f"{BASE_URL}/search")
    assert response.status_code == 422