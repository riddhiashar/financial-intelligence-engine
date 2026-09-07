# 📈 Financial Intelligence & Alerting Engine

A production-grade AI pipeline that processes financial news headlines,
classifies market sentiment, enables semantic search across historical data,
and generates concise market intelligence briefings using a local LLM.

Built as an 8-week internship project covering the full ML engineering stack —
from raw data ingestion to containerized deployment.

---

## 🏗️ Architecture
┌─────────────────────────────────┐
                    │         User / Browser          │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │     Streamlit Dashboard         │
                    │        localhost:8501           │
                    └────────────┬────────────────────┘
                                 │ HTTP REST
                    ┌────────────▼────────────────────┐
                    │       FastAPI Backend           │
                    │        localhost:8000           │
                    └──────┬──────────────┬───────────┘
                           │              │
          ┌────────────────▼───┐    ┌─────▼──────────────────┐
          │   Qdrant Vector DB │    │   Ollama (Llama3 LLM)  │
          │    localhost:6333  │    │    localhost:11434      │
          │   9,380 vectors    │    │     4.7GB model         │
          └────────────────────┘    └────────────────────────┘
          ---

## 🚀 Quick Start — Docker (Recommended)

### Prerequisites
- Docker Desktop installed and running
- Ollama installed with Llama3 pulled

```bash
# Pull Llama3 model (one time only)
ollama pull llama3

# Start Ollama
ollama serve
```

### Start Everything

```bash
# Clone and navigate to project
cd financial-intelligence-engine

# Copy environment file
cp .env.example .env

# Build and start all services
docker compose up --build
```

### Access the Application

| Service            | URL                          |
|--------------------|------------------------------|
| Streamlit Dashboard| http://localhost:8501        |
| FastAPI Swagger UI | http://localhost:8000/docs   |
| Qdrant Dashboard   | http://localhost:6333/dashboard |

### Stop Everything

```bash
docker compose down
```

### Stop and Remove All Data

```bash
docker compose down -v
```

---

## 💻 Local Development (Without Docker)

### Prerequisites

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r notebook/requirements.txt
```

### Start Services

```bash
# Terminal 1 — Qdrant
docker run -d --name qdrant_financial \
  -p 6333:6333 -p 6334:6334 \
  qdrant/qdrant

# Terminal 2 — Ollama
ollama serve

# Terminal 3 — FastAPI
cd notebook
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 4 — Streamlit
cd notebook
streamlit run app.py
```

---

## 📁 Folder Structure
financial-intelligence-engine/
│
├── Dockerfile.api # FastAPI container
├── Dockerfile.streamlit # Streamlit container
├── docker-compose.yml # Multi-container orchestration
├── .env.example # Environment variables template
├── .dockerignore # Docker build exclusions
├── README.md # This file
│
└── notebook/ # All application code
├── main.py # FastAPI application
├── app.py # Streamlit dashboard
├── predict.py # DistilBERT sentiment prediction
├── vector_db.py # Qdrant vector DB operations
├── embedding_pipeline.py# Sentence embedding utilities
├── semantic_search.py # Cosine similarity search
├── metadata_pipeline.py # Metadata extraction utilities
├── rag_pipeline.py # LangChain RAG pipeline
├── ingest.py # HuggingFace data ingestion
├── test_api.py # FastAPI automated tests
├── test_predict.py # Prediction module tests
├── test_rag.py # RAG pipeline tests
├── performance_test.py # API benchmarking
├── requirements.txt # Python dependencies
│
├── distilbert_best_model.pt # Fine-tuned model weights
├── distilbert_tokenizer/ # Saved tokenizer
├── headline_embeddings.npy # 9380 × 384 embeddings
├── metadata.json # Headline metadata
└── day4_verified.csv # Cleaned dataset


---

## 🔌 API Endpoints

| Method | Endpoint         | Description                                      |
|--------|------------------|--------------------------------------------------|
| GET    | `/health`        | System health check — API, Qdrant, model status  |
| POST   | `/analyze`       | Sentiment analysis for a single headline         |
| POST   | `/batch`         | Sentiment analysis for up to 50 headlines        |
| GET    | `/search`        | Semantic similarity search                       |
| GET    | `/hybrid-search` | Ticker-aware search with keyword boosting        |
| POST   | `/summarize`     | RAG market intelligence summary via Llama3       |
| GET    | `/stats`         | Dataset statistics                               |

Full interactive documentation: **http://localhost:8000/docs**

---

## 🤖 Models

| Model                        | Purpose              | Size   | Performance           |
|------------------------------|----------------------|--------|-----------------------|
| `distilbert-base-uncased`    | Sentiment classifier | 268MB  | 84.43% accuracy       |
| `all-MiniLM-L6-v2`          | Text embeddings      | ~90MB  | 384 dimensions        |
| `llama3` (via Ollama)        | RAG summarization    | 4.7GB  | 2-sentence briefings  |

---

## 📊 Dataset

- **Source:** `zeroshot/twitter-financial-news-sentiment` (HuggingFace)
- **Raw rows:** 9,543
- **Clean rows:** 9,380
- **Classes:** Bearish 15.2% · Bullish 20.3% · Neutral 64.5%
- **Imbalance handling:** SMOTE (TF-IDF) · Class weights (DistilBERT)

---

## 🛠️ Tech Stack

| Layer          | Technology                          | Version    |
|----------------|-------------------------------------|------------|
| Language       | Python                              | 3.13       |
| ML Framework   | PyTorch + HuggingFace Transformers  | 2.1 / 5.13 |
| Embedding      | Sentence Transformers               | 5.6.0      |
| LLM            | Ollama + Llama3                     | 0.32.5     |
| LLM Orchestration | LangChain                        | 1.3.14     |
| Vector DB      | Qdrant                              | latest     |
| API Framework  | FastAPI + Uvicorn                   | 0.115 / 0.34 |
| Dashboard      | Streamlit                           | 1.55.0     |
| Visualization  | Plotly                              | 5.24.1     |
| Containerization | Docker + Docker Compose           | 29.6 / v5.3 |

---

## 🏥 Health Checks

| Service    | Endpoint                    | Interval | Retries |
|------------|-----------------------------|----------|---------|
| Qdrant     | `http://qdrant:6333/healthz`| 10s      | 5       |
| FastAPI    | `http://api:8000/health`    | 30s      | 5       |
| Streamlit  | `http://streamlit:8501/_stcore/health` | 30s | 3   |

Docker Compose starts services in dependency order:

Qdrant → (healthy) → FastAPI → (healthy) → Streamlit


---

## 🧪 Running Tests

```bash
# Unit tests for prediction module
python test_predict.py

# API endpoint tests (FastAPI must be running)
pytest test_api.py -v

# RAG pipeline tests (FastAPI + Ollama must be running)
pytest test_rag.py -v

# Performance benchmarks (FastAPI must be running)
python performance_test.py
```

---

## 📈 Model Performance

### DistilBERT — Test Set Results

| Class    | Precision | Recall | F1   |
|----------|-----------|--------|------|
| Bearish  | 0.66      | 0.84   | 0.74 |
| Bullish  | 0.80      | 0.74   | 0.77 |
| Neutral  | 0.92      | 0.88   | 0.90 |
| **Macro**| **0.79**  | **0.82**| **0.80** |

**Overall Accuracy: 84.43%** vs TF-IDF baseline: 76.87%

### API Performance Benchmarks

| Endpoint                    | Mean Latency |
|-----------------------------|--------------|
| GET /health                 | ~18ms        |
| POST /analyze (no similar)  | ~21ms        |
| POST /analyze (with similar)| ~72ms        |
| POST /batch (3 items)       | ~57ms        |
| GET /search                 | ~25ms        |
| GET /hybrid-search          | ~35ms        |
| POST /summarize (LLM)       | 10-40s (CPU) |

*Benchmarked using session-based connection reuse after TCP warmup.*

---

## ⚙️ Environment Variables

| Variable      | Default                   | Description                        |
|---------------|---------------------------|------------------------------------|
| `QDRANT_HOST` | `localhost`               | Qdrant host (`qdrant` in Docker)   |
| `OLLAMA_HOST` | `localhost`               | Ollama host (`host.docker.internal` in Docker) |
| `API_BASE_URL`| `http://localhost:8000`   | FastAPI URL for Streamlit          |

Copy `.env.example` to `.env` and update values for your environment.

---

## 📋 Weekly Progress

| Week | Focus                        | Deliverable                          |
|------|------------------------------|--------------------------------------|
| 1    | Data Pipeline & EDA          | Cleaned dataset, ingestion script    |
| 2    | Sentiment Classification     | DistilBERT model, predict.py         |
| 3    | Embedding Pipeline           | 9380 embeddings, semantic search     |
| 4    | Vector Database              | Qdrant collection, vector_db.py      |
| 5    | FastAPI Backend              | REST API, 35 automated tests         |
| 6    | LLM Integration & RAG        | Llama3 RAG pipeline, /summarize      |
| 7    | Streamlit Dashboard          | Full UI with hybrid search           |
| 8    | Docker Containerization      | Single-command deployment            |


## Repository Notes

To keep the repository lightweight and within GitHub's file size limits, some large machine learning artifacts and datasets are excluded using `.gitignore`.

The following files/directories are **not included** in this repository:

- `notebook/distilbert_best_model.pt`
- `notebook/headline_embeddings.npy`
- `notebook/distilbert_tokenizer/`
- `notebook/metadata.json`
- `notebook/financial_news.csv`
- `notebook/day3_preprocessed.csv`
- `notebook/day4_verified.csv`

These files are required to run the project locally and should be placed in the `notebook/` directory before starting the application.

The source code, Docker configuration, API implementation, Streamlit dashboard, and deployment setup are fully included in this repository.



