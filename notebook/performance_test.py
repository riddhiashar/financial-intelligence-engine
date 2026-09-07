"""
performance_test.py — API Performance Benchmarking
Week 5 Day 4 (Fixed Version)

Uses session-based connection reuse for accurate latency measurement.
Run with: python performance_test.py
Make sure API is running first:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import time
import json
import requests
import statistics

BASE_URL = "http://localhost:8000"

# use session for connection reuse — eliminates TCP overhead
session = requests.Session()


def warmup():
    """Warm up connection and model caches before benchmarking."""
    print("Warming up API connections and model caches...")
    try:
        # warmup requests — not measured
        for _ in range(3):
            session.get(f"{BASE_URL}/health", timeout=30)
            session.post(
                f"{BASE_URL}/analyze",
                json={"text": "$AAPL beats earnings", "include_similar": False},
                timeout=30
            )
        print("Warmup complete\n")
    except Exception as e:
        print(f"Warmup failed: {e}")


def measure_endpoint(name, method, url, payload=None,
                     params=None, runs=10):
    """
    Measure endpoint latency over multiple runs.
    Uses persistent session and excludes first run from stats.
    """
    latencies = []
    errors    = 0

    for i in range(runs):
        try:
            start = time.perf_counter()
            if method == "GET":
                response = session.get(
                    url, params=params, timeout=30
                )
            else:
                response = session.post(
                    url, json=payload, timeout=30
                )
            latency = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                latencies.append(latency)
            else:
                errors += 1
        except Exception as e:
            errors += 1

    if not latencies:
        return None

    sorted_lats = sorted(latencies)
    p95_index   = max(0, int(0.95 * len(latencies)) - 1)

    return {
        "endpoint":  name,
        "runs":      runs,
        "errors":    errors,
        "min_ms":    round(min(latencies), 2),
        "max_ms":    round(max(latencies), 2),
        "mean_ms":   round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms":    round(sorted_lats[p95_index], 2),
    }


def run_benchmarks():
    print("=" * 65)
    print("Financial Intelligence API — Performance Benchmark")
    print("=" * 65)
    print(f"Base URL: {BASE_URL}")
    print(f"Runs per endpoint: 10")
    print(f"Method: Session-based connection reuse\n")

    # warmup first
    warmup()

    results = []

    # 1. Health check
    print("Benchmarking GET /health...")
    r = measure_endpoint(
        name   = "GET /health",
        method = "GET",
        url    = f"{BASE_URL}/health"
    )
    if r:
        results.append(r)
        print(f"  Mean: {r['mean_ms']}ms  "
              f"Median: {r['median_ms']}ms  "
              f"P95: {r['p95_ms']}ms\n")

    # 2. Analyze — bullish no similar
    print("Benchmarking POST /analyze (no similar search)...")
    r = measure_endpoint(
        name    = "POST /analyze (no similar)",
        method  = "POST",
        url     = f"{BASE_URL}/analyze",
        payload = {
            "text":            "$AAPL beats earnings estimates",
            "include_similar": False
        }
    )
    if r:
        results.append(r)
        print(f"  Mean: {r['mean_ms']}ms  "
              f"Median: {r['median_ms']}ms  "
              f"P95: {r['p95_ms']}ms\n")

    # 3. Analyze — with similar search
    print("Benchmarking POST /analyze (with similar search)...")
    r = measure_endpoint(
        name    = "POST /analyze (with similar)",
        method  = "POST",
        url     = f"{BASE_URL}/analyze",
        payload = {
            "text":            "$AAPL beats earnings estimates revenue up 12%",
            "include_similar": True,
            "similar_limit":   5
        }
    )
    if r:
        results.append(r)
        print(f"  Mean: {r['mean_ms']}ms  "
              f"Median: {r['median_ms']}ms  "
              f"P95: {r['p95_ms']}ms\n")

    # 4. Batch — 3 headlines
    print("Benchmarking POST /batch (3 headlines)...")
    r = measure_endpoint(
        name    = "POST /batch (3 items)",
        method  = "POST",
        url     = f"{BASE_URL}/batch",
        payload = {
            "texts": [
                "$AAPL beats earnings estimates revenue up 12%",
                "$TSLA misses revenue target stock falls",
                "Federal Reserve announces rate decision"
            ]
        }
    )
    if r:
        results.append(r)
        print(f"  Mean: {r['mean_ms']}ms  "
              f"Median: {r['median_ms']}ms  "
              f"P95: {r['p95_ms']}ms\n")

    # 5. Batch — 10 headlines
    print("Benchmarking POST /batch (10 headlines)...")
    r = measure_endpoint(
        name    = "POST /batch (10 items)",
        method  = "POST",
        url     = f"{BASE_URL}/batch",
        payload = {
            "texts": [
                "$AAPL beats earnings estimates",
                "$TSLA misses revenue target",
                "Federal Reserve rate decision",
                "$MSFT price target raised analysts",
                "Oil prices drop demand concerns",
                "$NVDA record revenue smashing estimates",
                "$META shares plunge user growth",
                "Apple scheduled quarterly earnings",
                "$AMZN beats revenue misses EPS",
                "US inflation drops lowest level"
            ]
        }
    )
    if r:
        results.append(r)
        print(f"  Mean: {r['mean_ms']}ms  "
              f"Median: {r['median_ms']}ms  "
              f"P95: {r['p95_ms']}ms\n")

    # 6. Search — no filter
    print("Benchmarking GET /search (no filter)...")
    r = measure_endpoint(
        name   = "GET /search (no filter)",
        method = "GET",
        url    = f"{BASE_URL}/search",
        params = {"query": "Tesla beats earnings", "limit": 5}
    )
    if r:
        results.append(r)
        print(f"  Mean: {r['mean_ms']}ms  "
              f"Median: {r['median_ms']}ms  "
              f"P95: {r['p95_ms']}ms\n")

    # 7. Search — sentiment filter
    print("Benchmarking GET /search (Bearish filter)...")
    r = measure_endpoint(
        name   = "GET /search (Bearish filter)",
        method = "GET",
        url    = f"{BASE_URL}/search",
        params = {
            "query":     "stock market crash recession",
            "sentiment": "Bearish",
            "limit":     5
        }
    )
    if r:
        results.append(r)
        print(f"  Mean: {r['mean_ms']}ms  "
              f"Median: {r['median_ms']}ms  "
              f"P95: {r['p95_ms']}ms\n")

    # 8. Stats
    print("Benchmarking GET /stats...")
    r = measure_endpoint(
        name   = "GET /stats",
        method = "GET",
        url    = f"{BASE_URL}/stats"
    )
    if r:
        results.append(r)
        print(f"  Mean: {r['mean_ms']}ms  "
              f"Median: {r['median_ms']}ms  "
              f"P95: {r['p95_ms']}ms\n")

    # summary table
    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)
    print(
        f"{'Endpoint':<35} {'Mean':>8} {'Median':>8} "
        f"{'P95':>8} {'Min':>8}"
    )
    print("-" * 65)
    for r in results:
        print(
            f"{r['endpoint']:<35} "
            f"{r['mean_ms']:>7}ms "
            f"{r['median_ms']:>7}ms "
            f"{r['p95_ms']:>7}ms "
            f"{r['min_ms']:>7}ms"
        )

    # save results
    with open("week5_performance.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: week5_performance.json")

    # recommendations
    print("\n" + "=" * 65)
    print("PERFORMANCE ANALYSIS")
    print("=" * 65)
    for r in results:
        mean = r["mean_ms"]
        if mean > 1000:
            status = "SLOW"
            tip    = "Consider model caching or GPU acceleration"
        elif mean > 500:
            status = "MODERATE"
            tip    = "Acceptable for CPU inference"
        elif mean > 100:
            status = "GOOD"
            tip    = "Good response time for CPU deployment"
        else:
            status = "FAST"
            tip    = "Excellent response time"
        print(f"{status}: {r['endpoint']} — {mean}ms")
        print(f"  {tip}")

    # close session
    session.close()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    run_benchmarks()