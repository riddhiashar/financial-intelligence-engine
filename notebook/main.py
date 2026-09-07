"""
main.py — Financial Intelligence & Alerting Engine
Week 5 Day 2: Input Validation + Error Handling + Request Logging

Endpoints:
    GET  /health              → health check
    POST /analyze             → sentiment analysis + similar headlines
    POST /batch               → batch sentiment analysis
    GET  /search              → semantic search for similar headlines

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import uuid
import logging
import numpy as np



# add these two constants — replace your existing hardcoded values
QDRANT_HOST  = os.getenv("QDRANT_HOST",  "localhost")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "localhost")

from vector_db import get_client, qdrant_search, ticker_aware_search, hybrid_search

from qdrant_client.models import Filter, FieldCondition, MatchValue
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
VECTOR_SIZE = 384

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

from predict import predict, batch_predict, full_clean_pipeline
from vector_db import get_client, qdrant_search
from embedding_pipeline import embed_text


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Intelligence & Alerting Engine",
    description="""
Production-grade API for financial news sentiment analysis.

## Endpoints
- **POST /analyze** — Analyze sentiment of a single financial headline
- **POST /batch**   — Analyze sentiment of multiple headlines at once
- **GET  /search**  — Search for similar historical headlines
- **GET  /health**  — System health check

## Sentiment Classes
- **Bearish** — Negative market sentiment (label: 0)
- **Bullish** — Positive market sentiment (label: 1)
- **Neutral** — No clear directional sentiment (label: 2)
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

COLLECTION_NAME  = "financial_headlines"
MAX_TEXT_LENGTH  = 500
MAX_BATCH_SIZE   = 50
VALID_SENTIMENTS = {"Bearish", "Bullish", "Neutral"}
OLLAMA_MODEL = "llama3"
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





qdrant_client = None
llm         = None
rag_chain   = None



@app.on_event("startup")
async def startup_event():
    global qdrant_client, llm, rag_chain
    logger.info("Starting Financial Intelligence API...")

    # connect Qdrant using environment variable
    try:
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:6333...")
        qdrant_client = get_client(host=QDRANT_HOST, port=6333)
        logger.info("Qdrant connected successfully")
    except Exception as e:
        logger.error(f"Qdrant connection failed: {e}")

    # load LLM using environment variable
    try:
        logger.info(f"Loading {OLLAMA_MODEL} via Ollama at {OLLAMA_HOST}:11434...")
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
        prompt_template = PromptTemplate(
            input_variables=["context", "num_headlines"],
            template=RAG_TEMPLATE
        )
        rag_chain = prompt_template | llm
        logger.info("LLM and RAG chain loaded successfully")
    except Exception as e:
        logger.error(f"LLM loading failed: {e}")

    logger.info("API startup complete")


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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(
        f"[{request_id}] {request.method} {request.url.path} — started"
    )

    response = await call_next(request)

    duration = (time.time() - start_time) * 1000
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"— {response.status_code} — {duration:.1f}ms"
    )

    response.headers["X-Request-ID"]    = request_id
    response.headers["X-Response-Time"] = f"{duration:.1f}ms"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error":   "Internal server error",
            "detail":  str(exc),
            "path":    str(request.url.path),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Raw financial headline to analyze"
    )
    include_similar: Optional[bool] = Field(
        default=True,
        description="Whether to include similar historical headlines"
    )
    similar_limit: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar headlines to return (1-20)"
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "text": "$AAPL beats earnings estimates revenue up 12%",
                "include_similar": True,
                "similar_limit": 5
            }
        }


class BatchRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of raw financial headlines to analyze"
    )
    include_similar: Optional[bool] = Field(
        default=False,
        description="Include similar headlines for each text"
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        if not v:
            raise ValueError("texts list cannot be empty")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Maximum batch size is {MAX_BATCH_SIZE}. "
                f"Got {len(v)} texts."
            )
        cleaned = []
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string")
            if not text.strip():
                raise ValueError(
                    f"Text at index {i} cannot be empty or whitespace"
                )
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"Text at index {i} exceeds maximum length "
                    f"of {MAX_TEXT_LENGTH} characters"
                )
            cleaned.append(text.strip())
        return cleaned

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "$AAPL beats earnings estimates revenue up 12%",
                    "$TSLA misses revenue target stock falls after hours",
                    "Federal Reserve announces interest rate decision"
                ],
                "include_similar": False
            }
        }

# ─── Response Models ──────────────────────────────────────
class SimilarHeadline(BaseModel):
    rank:       int
    similarity: float
    headline:   str
    sentiment:  str
    ticker:     str

class AnalyzeResponse(BaseModel):
    request_id:         str
    timestamp:          str
    original_text:      str
    cleaned_text:       str
    sentiment:          str
    label:              int
    confidence:         str
    scores:             dict
    similar_headlines:  List[SimilarHeadline]
    processing_time_ms: float

class BatchResultItem(BaseModel):
    index:      int
    original_text: str
    cleaned_text:  str
    sentiment:  str
    label:      int
    confidence: str
    scores:     dict
    error:      Optional[str] = None

class BatchResponse(BaseModel):
    request_id:         str
    timestamp:          str
    total:              int
    successful:         int
    failed:             int
    results:            List[BatchResultItem]
    processing_time_ms: float

class SearchResult(BaseModel):
    rank:       int
    similarity: float
    headline:   str
    sentiment:  str
    ticker:     str
    id:         int

class SummarizeRequest(BaseModel):
    query:            str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Query to search and summarize"
    )
    top_k:            Optional[int]  = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of headlines to use as context (1-10)"
    )
    sentiment_filter: Optional[str]  = Field(
        default=None,
        description="Filter context by sentiment: Bearish, Bullish, Neutral"
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()

    @field_validator("sentiment_filter")
    @classmethod
    def validate_sentiment(cls, v):
        if v and v not in {"Bearish", "Bullish", "Neutral"}:
            raise ValueError(
                "sentiment_filter must be Bearish, Bullish or Neutral"
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query":            "Tesla earnings revenue performance",
                "top_k":            5,
                "sentiment_filter": None
            }
        }


class SummarizeResponse(BaseModel):
    request_id:          str
    timestamp:           str
    query:               str
    summary:             str
    sentiment_filter:    Optional[str]
    sources:             List[SimilarHeadline]
    retrieval_time_ms:   float
    llm_time_ms:         float
    processing_time_ms:  float


class HealthResponse(BaseModel):
    status:     str
    qdrant:     str
    model:      str
    collection: str
    version:    str
    timestamp:  str


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"]
)
async def health_check():
    """
    Check if the API and all dependencies are running correctly.
    Returns status of Qdrant connection and loaded models.
    """
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        qdrant_status = (
            "connected"
            if COLLECTION_NAME in collection_names
            else "collection missing"
        )
    except Exception as e:
        qdrant_status = f"error: {str(e)}"

    return HealthResponse(
        status     = "ok",
        qdrant     = qdrant_status,
        model      = "distilbert-base-uncased (fine-tuned)",
        collection = COLLECTION_NAME,
        version    = "1.0.0",
        timestamp  = datetime.utcnow().isoformat()
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze Single Headline",
    tags=["Analysis"]
)
async def analyze(request: AnalyzeRequest):
    """
    Analyze the sentiment of a single financial headline.

    **Process:**
    1. Validates and cleans input text
    2. Predicts sentiment using fine-tuned DistilBERT
    3. Returns Bearish, Bullish or Neutral with confidence score
    4. Optionally returns similar historical headlines from Qdrant

    **Error codes:**
    - 400: Invalid input (empty text, text too long)
    - 500: Internal processing error
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    logger.info(
        f"[{request_id}] Analyzing: {request.text[:50]}..."
    )

    # predict sentiment
    try:
        prediction = predict(request.text)
    except Exception as e:
        logger.error(f"[{request_id}] Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    if "error" in prediction:
        raise HTTPException(
            status_code=400,
            detail=prediction["error"]
        )

    # find similar headlines
    similar = []
    if request.include_similar:
        try:
            search_results = hybrid_search(
                client          = qdrant_client,
                query           = request.query,
                top_k           = request.top_k,
                min_similarity  = 0.3,
                sentiment_filter= request.sentiment_filter,
                collection_name = COLLECTION_NAME
            )
            similar = [
                SimilarHeadline(
                    rank       = r["rank"],
                    similarity = round(r["similarity"], 4),
                    headline   = r["headline"],
                    sentiment  = r["sentiment"],
                    ticker     = r["ticker"] or "None"
                )
                for r in search_results
            ]
        except Exception as e:
            logger.warning(
                f"[{request_id}] Similar search failed: {e}"
            )
            similar = []

    processing_time = (time.time() - start_time) * 1000
    logger.info(
        f"[{request_id}] Completed in {processing_time:.1f}ms "
        f"— {prediction['sentiment']}"
    )

    return AnalyzeResponse(
        request_id         = request_id,
        timestamp          = datetime.utcnow().isoformat(),
        original_text      = prediction["original_text"],
        cleaned_text       = prediction["cleaned_text"],
        sentiment          = prediction["sentiment"],
        label              = prediction["label"],
        confidence         = prediction["confidence"],
        scores             = prediction["scores"],
        similar_headlines  = similar,
        processing_time_ms = round(processing_time, 2)
    )


@app.post(
    "/batch",
    response_model=BatchResponse,
    summary="Analyze Multiple Headlines",
    tags=["Analysis"]
)
async def batch_analyze(request: BatchRequest):
    """
    Analyze sentiment for multiple financial headlines at once.

    **More efficient than calling /analyze repeatedly.**

    **Limits:**
    - Maximum 50 headlines per request
    - Each headline max 500 characters

    **Error codes:**
    - 400: Invalid input (empty list, too many items, empty text)
    - 500: Internal processing error
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    logger.info(
        f"[{request_id}] Batch analyzing {len(request.texts)} headlines"
    )

    results     = []
    successful  = 0
    failed      = 0

    for i, text in enumerate(request.texts):
        try:
            prediction = predict(text)
            if "error" in prediction:
                results.append(BatchResultItem(
                    index        = i,
                    original_text= text,
                    cleaned_text = "",
                    sentiment    = "",
                    label        = -1,
                    confidence   = "0%",
                    scores       = {},
                    error        = prediction["error"]
                ))
                failed += 1
            else:
                results.append(BatchResultItem(
                    index        = i,
                    original_text= prediction["original_text"],
                    cleaned_text = prediction["cleaned_text"],
                    sentiment    = prediction["sentiment"],
                    label        = prediction["label"],
                    confidence   = prediction["confidence"],
                    scores       = prediction["scores"],
                    error        = None
                ))
                successful += 1
        except Exception as e:
            results.append(BatchResultItem(
                index        = i,
                original_text= text,
                cleaned_text = "",
                sentiment    = "",
                label        = -1,
                confidence   = "0%",
                scores       = {},
                error        = str(e)
            ))
            failed += 1

    processing_time = (time.time() - start_time) * 1000
    logger.info(
        f"[{request_id}] Batch complete — "
        f"{successful} success, {failed} failed — "
        f"{processing_time:.1f}ms"
    )

    return BatchResponse(
        request_id         = request_id,
        timestamp          = datetime.utcnow().isoformat(),
        total              = len(request.texts),
        successful         = successful,
        failed             = failed,
        results            = results,
        processing_time_ms = round(processing_time, 2)
    )


@app.get(
    "/search",
    response_model=List[SearchResult],
    summary="Semantic Search",
    tags=["Search"]
)
async def search(
    query: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description="Search query — e.g. 'Tesla beats earnings'"
    ),
    limit: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (1-20)"
    ),
    sentiment: Optional[str] = Query(
        default=None,
        description="Filter by sentiment: Bearish, Bullish or Neutral"
    ),
    ticker: Optional[str] = Query(
        default=None,
        description="Filter by ticker symbol e.g. $AAPL"
    ),
    min_similarity: float = Query(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0-1.0)"
    )
):
    """
    Search for similar historical financial headlines using semantic search.

    **Process:**
    1. Converts query to 384-dimensional embedding vector
    2. Searches Qdrant HNSW index for nearest neighbors
    3. Returns top-k most similar headlines with scores

    **Filters:**
    - sentiment: Bearish, Bullish or Neutral
    - ticker: e.g. $AAPL, $TSLA
    - min_similarity: filter out low quality matches

    **Error codes:**
    - 400: Invalid query or sentiment value
    - 500: Search engine error
    """
    # validate sentiment
    if sentiment and sentiment not in VALID_SENTIMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sentiment '{sentiment}'. "
                   f"Choose from: Bearish, Bullish, Neutral"
        )
   
    if ticker and not ticker.startswith("$"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker format '{ticker}'. "
                   f"Ticker must start with $ e.g. $AAPL"
        )

    try:
        results = qdrant_search(
            client           = qdrant_client,
            query            = query,
            top_k            = limit,
            min_similarity   = min_similarity,
            sentiment_filter = sentiment,
            ticker_filter    = ticker,
            collection_name  = COLLECTION_NAME
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

    if not results:
        return []

    return [
        SearchResult(
            rank       = r["rank"],
            similarity = round(r["similarity"], 4),
            headline   = r["headline"],
            sentiment  = r["sentiment"],
            ticker     = r["ticker"] or "None",
            id         = r["id"]
        )
        for r in results
    ]

@app.get(
    "/stats",
    summary="Dataset Statistics",
    tags=["System"]
)
async def get_stats():
    """
    Returns statistics about the financial headlines dataset
    stored in the vector database.
    """
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        total_points    = collection_info.points_count

        # count by sentiment using scroll
        bearish_count = len(qdrant_client.scroll(
            collection_name = COLLECTION_NAME,
            scroll_filter   = Filter(must=[
                FieldCondition(
                    key   = "sentiment",
                    match = MatchValue(value="Bearish")
                )
            ]),
            limit = 1
        )[0])

        bullish_count = len(qdrant_client.scroll(
            collection_name = COLLECTION_NAME,
            scroll_filter   = Filter(must=[
                FieldCondition(
                    key   = "sentiment",
                    match = MatchValue(value="Bullish")
                )
            ]),
            limit = 1
        )[0])

        return {
            "total_headlines":  total_points,
            "collection":       COLLECTION_NAME,
            "vector_dimensions": VECTOR_SIZE,
            "sentiment_distribution": {
                "Bearish": 1428,
                "Bullish": 1904,
                "Neutral": 6048
            },
            "models": {
                "sentiment":  "distilbert-base-uncased (fine-tuned)",
                "embeddings": "all-MiniLM-L6-v2 (384 dims)"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Stats retrieval failed: {str(e)}"
        )

@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="RAG Market Intelligence Summary",
    tags=["RAG"]
)
async def summarize(request: SummarizeRequest):
    """
    Generate a 2-sentence market intelligence briefing using RAG.

    **Process:**
    1. Search Qdrant for top-k similar financial headlines
    2. Use retrieved headlines as context for Llama3
    3. Generate a concise 2-sentence market intelligence briefing

    **Filters:**
    - sentiment_filter: restrict context to Bearish, Bullish or Neutral

    **Error codes:**
    - 400: Invalid input
    - 503: LLM not available (Ollama not running)
    - 500: Internal error
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    logger.info(
        f"[{request_id}] Summarize: {request.query[:50]}..."
    )

    # check LLM is loaded
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="LLM not available. Make sure Ollama is running "
                   "with llama3 model pulled."
        )

    # validate sentiment filter
    if request.sentiment_filter and \
       request.sentiment_filter not in VALID_SENTIMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sentiment_filter. "
                   f"Choose from: Bearish, Bullish, Neutral"
        )

    # step 1: retrieve similar headlines
    retrieval_start = time.time()
    try:
        search_results = qdrant_search(
            client           = qdrant_client,
            query            = request.query,
            top_k            = request.top_k,
            min_similarity   = 0.5,
            sentiment_filter = request.sentiment_filter,
            collection_name  = COLLECTION_NAME
        )
    except Exception as e:
        logger.error(f"[{request_id}] Retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Headline retrieval failed: {str(e)}"
        )

    retrieval_time = (time.time() - retrieval_start) * 1000

    if not search_results:
        return SummarizeResponse(
            request_id         = request_id,
            timestamp          = datetime.utcnow().isoformat(),
            query              = request.query,
            summary            = "No relevant headlines found "
                                 "for this query.",
            sentiment_filter   = request.sentiment_filter,
            sources            = [],
            retrieval_time_ms  = round(retrieval_time, 2),
            llm_time_ms        = 0.0,
            processing_time_ms = round(
                (time.time() - start_time) * 1000, 2
            )
        )

    # step 2: build context
    context = "\n".join([
        f"{i+1}. [{r['sentiment']}] {r['headline']} "
        f"(similarity: {r['similarity']:.2f})"
        for i, r in enumerate(search_results)
    ])

    # step 3: generate summary
    llm_start = time.time()
    try:
        raw_summary = rag_chain.invoke({
            "context":       context,
            "num_headlines": len(search_results)
        })
        summary = clean_summary(raw_summary)
    except Exception as e:
        logger.error(f"[{request_id}] LLM generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Summary generation failed: {str(e)}"
        )

    llm_time       = (time.time() - llm_start) * 1000
    total_time     = (time.time() - start_time) * 1000

    logger.info(
        f"[{request_id}] Summary generated in {total_time:.1f}ms"
    )

    return SummarizeResponse(
        request_id         = request_id,
        timestamp          = datetime.utcnow().isoformat(),
        query              = request.query,
        summary            = summary,
        sentiment_filter   = request.sentiment_filter,
        sources            = [
            SimilarHeadline(
                rank       = r["rank"],
                similarity = round(r["similarity"], 4),
                headline   = r["headline"],
                sentiment  = r["sentiment"],
                ticker     = r["ticker"] or "None"
            )
            for r in search_results
        ],
        retrieval_time_ms  = round(retrieval_time, 2),
        llm_time_ms        = round(llm_time, 2),
        processing_time_ms = round(total_time, 2)
    )
class HybridSearchResult(BaseModel):
    rank:           int
    similarity:     float
    hybrid_score:   float
    keyword_boost:  float
    headline:       str
    sentiment:      str
    ticker:         str
    id:             int
    retrieval_type: str


@app.get(
    "/hybrid-search",
    response_model=List[HybridSearchResult],
    summary="Hybrid Semantic + Keyword Search",
    tags=["Search"]
)
async def hybrid_search_endpoint(
    query: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description="Search query — ticker symbols like $AAPL get prioritized"
    ),
    limit: int = Query(
        default=5, ge=1, le=20,
        description="Number of results (1-20)"
    ),
    sentiment: Optional[str] = Query(
        default=None,
        description="Filter: Bearish, Bullish or Neutral"
    ),
    min_similarity: float = Query(
        default=0.3, ge=0.0, le=1.0,
        description="Minimum similarity threshold"
    )
):
    """
    Hybrid search combining semantic similarity + keyword boosting
    with ticker-aware retrieval.

    **Improvements over standard /search:**
    - Headlines containing $TICKER from query appear first
    - Keyword matches get a relevance boost
    - Better results for queries like '$AAPL iPhone revenue'

    **Returns hybrid_score** = similarity + keyword_boost
    """
    if sentiment and sentiment not in VALID_SENTIMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sentiment. "
                   f"Choose from: Bearish, Bullish, Neutral"
        )

    try:
        results = hybrid_search(
            client           = qdrant_client,
            query            = query,
            top_k            = limit,
            min_similarity   = min_similarity,
            sentiment_filter = sentiment,
            collection_name  = COLLECTION_NAME
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

    if not results:
        return []

    return [
        HybridSearchResult(
            rank           = r["rank"],
            similarity     = round(r["similarity"], 4),
            hybrid_score   = round(r.get("hybrid_score", r["similarity"]), 4),
            keyword_boost  = round(r.get("keyword_boost", 0.0), 4),
            headline       = r["headline"],
            sentiment      = r["sentiment"],
            ticker         = r["ticker"] or "None",
            id             = r["id"],
            retrieval_type = r.get("retrieval_type", "semantic")
        )
        for r in results
    ]



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host   = "0.0.0.0",
        port   = 8000,
        reload = True
    )
        