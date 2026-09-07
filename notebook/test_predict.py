"""
test_predict.py - Unit Tests for predict.py
Run with: pytest test_predict.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import predict, batch_predict, full_clean_pipeline

def test_predict_returns_correct_keys():
    result = predict("$AAPL beats earnings estimates")
    assert all(k in result for k in [
        "sentiment", "confidence", "scores", 
        "label", "cleaned_text", "original_text"
    ])

def test_bullish_headline():
    result = predict("$AAPL beats earnings estimates revenue up 12%")
    assert result["sentiment"] == "Bullish"

def test_bearish_headline():
    result = predict("$TSLA misses revenue target stock falls after hours")
    assert result["sentiment"] == "Bearish"

def test_neutral_headline():
    result = predict("Federal Reserve announces interest rate decision next week")
    assert result["sentiment"] == "Neutral"

def test_empty_string_no_crash():
    result = predict("")
    assert "error" in result

def test_none_input_no_crash():
    result = predict(None)
    assert "error" in result

def test_url_removed_by_cleaning():
    cleaned = full_clean_pipeline("$AAPL beats earnings https://t.co/abc123")
    assert "http" not in cleaned

def test_ticker_normalized_to_uppercase():
    cleaned = full_clean_pipeline("buying $aapl today")
    assert "$AAPL" in cleaned

def test_confidence_is_percentage_string():
    result = predict("$MSFT price target raised by analysts")
    assert "%" in result["confidence"]

def test_batch_predict_returns_correct_count():
    results = batch_predict([
        "$AAPL beats earnings",
        "$TSLA misses revenue",
        "Fed announces rate decision"
    ])
    assert len(results) == 3

if __name__ == "__main__":
    # run all tests manually without pytest
    tests = [
        test_predict_returns_correct_keys,
        test_bullish_headline,
        test_bearish_headline,
        test_neutral_headline,
        test_empty_string_no_crash,
        test_none_input_no_crash,
        test_url_removed_by_cleaning,
        test_ticker_normalized_to_uppercase,
        test_confidence_is_percentage_string,
        test_batch_predict_returns_correct_count,
    ]

    passed = 0
    failed = 0
    print("Running unit tests...\n")
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
