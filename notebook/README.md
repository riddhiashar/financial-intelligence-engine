# Financial Intelligence and Alerting Engine

A production-grade NLP pipeline that classifies financial news headlines
into Bearish, Bullish, or Neutral sentiment using DistilBERT.

## Project Structure

notebook/
    ingest.py                    data ingestion script
    predict.py                   production prediction module
    test_predict.py              unit tests
    Week2_Modeling.ipynb         modeling notebook
    distilbert_best_model.pt     trained DistilBERT model
    distilbert_tokenizer/        saved tokenizer
    tfidf_lr_model.pkl           baseline TF-IDF model
    tfidf_vectorizer.pkl         TF-IDF vectorizer
    financial_news.csv           raw dataset
    day4_verified.csv            cleaned verified dataset
    train_set.csv                training split
    val_set.csv                  validation split
    test_set.csv                 test split locked
    requirements.txt             dependencies

## Setup

Install dependencies
    pip install -r requirements.txt

## Usage

1. Run data ingestion pipeline
    python ingest.py
    Downloads dataset from HuggingFace, cleans it, saves to financial_news.csv

2. Run sentiment prediction
    from predict import predict, batch_predict

    single prediction
    result = predict("$AAPL beats earnings estimates revenue up 12%")
    print(result)

    batch prediction
    results = batch_predict([
        "$AAPL beats earnings estimates",
        "$TSLA misses revenue target",
        "Federal Reserve announces rate decision"
    ])

3. Run unit tests
    python test_predict.py
    Expected output: 10 passed, 0 failed

## Model Performance

DistilBERT Final Model Test Set Results
    Bearish  Precision 0.66  Recall 0.84  F1 0.74
    Bullish  Precision 0.80  Recall 0.74  F1 0.77
    Neutral  Precision 0.92  Recall 0.88  F1 0.90
    Macro    Precision 0.79  Recall 0.82  F1 0.80
    Overall Accuracy 84.43%

Baseline Comparison Test Set
    Overall Accuracy  TF-IDF 76.87%  DistilBERT 84.43%  Improvement +7.56%
    Bearish F1        TF-IDF 0.62    DistilBERT 0.74    Improvement +0.12
    Bullish F1        TF-IDF 0.64    DistilBERT 0.77    Improvement +0.13
    Neutral F1        TF-IDF 0.85    DistilBERT 0.90    Improvement +0.05
    Macro F1          TF-IDF 0.70    DistilBERT 0.80    Improvement +0.10

## Known Limitations

    Complex negation: not expected to miss predicted as Bearish
    Sarcasm: model does not detect sarcastic tone
    Macro indicators: inflation drops predicted as Bearish
    dropping inflation is actually Bullish for markets

## Tech Stack

    Python 3.13
    PyTorch 2.12
    HuggingFace Transformers 5.13
    DistilBERT distilbert-base-uncased
    Scikit-learn
    FastAPI Week 5
    Streamlit Week 7
    Docker Week 8

## Dataset

    Source: zeroshot/twitter-financial-news-sentiment HuggingFace
    Raw rows: 9543
    Clean rows: 9380
    Classes: Bearish 15.2% Bullish 20.3% Neutral 64.5%
    Imbalance handled with SMOTE for TF-IDF and Class weights for DistilBERT

    api_section = """
## API Usage (Week 5)

### Start the API
```bash
cd notebook
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

#### GET /health
Check system status.
```bash
curl http://localhost:8000/health
```

#### POST /analyze
Analyze sentiment of a financial headline.
```bash
curl -X POST http://localhost:8000/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"text": "$AAPL beats earnings estimates revenue up 12%"}'
```

#### POST /batch
Analyze multiple headlines at once.
```bash
curl -X POST http://localhost:8000/batch \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["$AAPL beats earnings", "$TSLA misses revenue"]}'
```

#### GET /search
Search for similar historical headlines.
```bash
curl "http://localhost:8000/search?query=Tesla+beats+earnings&limit=5"
```

#### GET /stats
Get dataset statistics.
```bash
curl http://localhost:8000/stats
```

### Run API Tests
```bash
pytest test_api.py -v
```

### Performance Benchmark
```bash
python performance_test.py
```
"""

with open("README.md", "a", encoding="utf-8") as f:
    f.write(api_section)

print("README.md updated with API documentation")
