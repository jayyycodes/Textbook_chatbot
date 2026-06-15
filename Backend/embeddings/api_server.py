#!/usr/bin/env python3
"""
FastAPI Persistent RAG Backend for LearnLens Textbook Chatbot
Loads all ML models (embedding, reranker) once at startup.
LLM priority: Groq → Gemini → Together.ai → OpenRouter

Usage:
    python api_server.py
    uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from cachetools import TTLCache
    _query_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)  # 1-hour TTL
except ImportError:
    _query_cache = {}

# Import pipeline components from search_faiss.py
from search_faiss import (
    SearchPipeline,
    AnswerGenerator,
    EmbeddingIndexer,
    CrossEncoderReranker,
    INDICES_DIR,
    TOP_K_FAISS,
    TOP_K_RERANK,
    BOOK_RELEVANCE_THRESHOLD,
    is_junk_chunk,
)

# Import LLM answer generation
try:
    from llm_answer import generate_answer, call_gemini, call_together_ai, call_openrouter, clean_answer
    from dotenv import load_dotenv
    load_dotenv()
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    textbook: str = Field(default="intro_ml", description="Textbook ID")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    textbook: str = Field(default="intro_ml", description="Textbook ID")


# =============================================================================
# GLOBAL STATE — Models loaded once at startup
# =============================================================================

loaded_pipelines: Dict[str, object] = {}
loaded_textbooks: Dict[str, dict] = {}
startup_time: float = 0
models_loaded: bool = False

# Request counters for /metrics
_request_counts: Dict[str, int] = defaultdict(int)
_total_latency: Dict[str, float] = defaultdict(float)


# =============================================================================
# TEXTBOOK ID MAPPING
# =============================================================================

TEXTBOOK_ALIASES = {
    'computer_networks': 'Computer_Networks',
    'ml': 'intro_to_ml',
    'machine_learning': 'intro_to_ml',
    'intro_ml': 'intro_to_ml',
    'intro_to_ml': 'intro_to_ml',
    'economics': 'economics',
}


def resolve_textbook_id(raw_id: str) -> str:
    """Resolve textbook aliases to canonical IDs."""
    return TEXTBOOK_ALIASES.get(raw_id.lower(), raw_id)


# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and indices at startup."""
    global startup_time, models_loaded

    logger.info("=" * 70)
    logger.info("🚀 STARTING FASTAPI RAG BACKEND")
    logger.info("=" * 70)

    start = time.time()

    # Discover all indexed textbooks
    config_files = list(INDICES_DIR.glob("*_config.json"))
    logger.info(f"Found {len(config_files)} textbook configs in {INDICES_DIR}")

    # Load each textbook pipeline independently
    # NOTE: We do NOT share embedders — each textbook may use a different model
    # (e.g. bge-large-en-v1.5 for Computer Networks vs all-MiniLM-L6-v2 for Economics)
    # Models load from local cache in ~100ms so the overhead is negligible.
    shared_reranker = None

    for config_path in config_files:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            textbook_id = config.get('textbook_id', config_path.stem.replace('_config', ''))
            textbook_name = config.get('textbook_name', textbook_id)

            logger.info(f"Loading: {textbook_name} ({textbook_id})...")

            pipeline = SearchPipeline(textbook_id=textbook_id)

            # Share only the reranker (same model for all textbooks, saves memory)
            if shared_reranker is None:
                shared_reranker = pipeline.reranker
            else:
                pipeline.reranker = shared_reranker

            loaded_pipelines[textbook_id] = pipeline
            loaded_textbooks[textbook_id] = {
                'id': textbook_id,
                'name': textbook_name,
                'description': config.get('description', ''),
                'model': config.get('model_name', 'unknown'),
                'total_chunks': config.get('total_chunks', 0),
                'embedding_dim': config.get('embedding_dim', 0),
            }

            logger.info(f"  ✓ Loaded {textbook_name}: {pipeline.index.ntotal} vectors")

        except Exception as e:
            logger.error(f"  ✗ Failed to load {config_path.name}: {e}")

    startup_time = time.time() - start
    models_loaded = True

    # Log LLM availability
    groq_ok      = bool(os.getenv('GROQ_API_KEY')) and os.getenv('GROQ_API_KEY') != 'your_groq_api_key_here'
    gemini_ok    = bool(os.getenv('GEMINI_API_KEY'))
    together_ok  = bool(os.getenv('TOGETHER_API_KEY'))
    openrouter_ok = bool(os.getenv('OPENROUTER_API_KEY'))
    logger.info(
        f"🤖 LLM providers: "
        f"Groq={'✓' if groq_ok else '✗'} | "
        f"Gemini={'✓' if gemini_ok else '✗'} | "
        f"Together={'✓' if together_ok else '✗'} | "
        f"OpenRouter={'✓' if openrouter_ok else '✗'}"
    )

    logger.info("=" * 70)
    logger.info(f"✅ ALL MODELS LOADED in {startup_time:.1f}s")
    logger.info(f"📚 Textbooks: {list(loaded_pipelines.keys())}")
    logger.info(f"🔗 API ready at http://0.0.0.0:8000")
    logger.info("=" * 70)

    yield  # App runs here

    # Shutdown
    logger.info("Shutting down FastAPI RAG Backend...")
    loaded_pipelines.clear()
    loaded_textbooks.clear()


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="LearnLens RAG API",
    description="Production RAG backend with Gemini-powered AI answers for textbook content",
    version="2.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    """Health check with system status."""
    return {
        "status": "healthy" if models_loaded else "loading",
        "models_loaded": models_loaded,
        "textbooks_loaded": len(loaded_pipelines),
        "textbook_ids": list(loaded_pipelines.keys()),
        "startup_time": f"{startup_time:.1f}s",
        "llm_available": HAS_LLM,
        "llm_providers": {
            "groq":       bool(os.getenv('GROQ_API_KEY')) and os.getenv('GROQ_API_KEY') != 'your_groq_api_key_here',
            "gemini":     bool(os.getenv('GEMINI_API_KEY')),
            "together":   bool(os.getenv('TOGETHER_API_KEY')),
            "openrouter": bool(os.getenv('OPENROUTER_API_KEY')),
        },
        "retrieval": "hybrid (BM25+FAISS+RRF)" if any(
            getattr(p, '_use_hybrid', False) for p in loaded_pipelines.values()
        ) else "dense-only",
    }


@app.get("/metrics")
async def metrics():
    """Request metrics for monitoring."""
    return {
        "requests": dict(_request_counts),
        "avg_latency_ms": {
            endpoint: round(_total_latency[endpoint] / max(1, _request_counts[endpoint]) * 1000, 1)
            for endpoint in _request_counts
        },
        "cache_size": len(_query_cache),
    }


@app.get("/textbooks")
async def list_textbooks():
    """List all available textbooks with metadata."""
    return {
        "textbooks": list(loaded_textbooks.values()),
        "total": len(loaded_textbooks),
    }


@app.post("/search")
async def search(request: SearchRequest):
    """
    Semantic search — returns reranked evidence chunks with match scores.
    No LLM rewriting, just direct textbook evidence.
    """
    start_time = time.time()
    textbook_id = resolve_textbook_id(request.textbook)
    if textbook_id not in loaded_pipelines:
        raise HTTPException(status_code=404, detail=f"Textbook '{request.textbook}' not found. Available: {list(loaded_pipelines.keys())}")
    pipeline = loaded_pipelines[textbook_id]
    try:
        result = pipeline.query(request.query, mode='raw')
        results = []
        rerank_scores = result.get('rerank_scores', {})
        if result.get('chunk_ids'):
            for chunk_id in result['chunk_ids']:
                for meta in pipeline.metadata:
                    if meta['chunk_id'] == chunk_id:
                        results.append({
                            'chunk_id': chunk_id,
                            'content': meta['text'][:800],
                            'word_count': meta.get('word_count', len(meta['text'].split())),
                            'rerank_score': rerank_scores.get(chunk_id, 0),
                        })
                        break
        duration = time.time() - start_time
        _request_counts['search'] += 1
        _total_latency['search'] += duration
        return {
            "query": request.query,
            "textbook": textbook_id,
            "textbook_name": loaded_textbooks.get(textbook_id, {}).get('name', textbook_id),
            "book_relevance": result.get('book_relevance', 0),
            "confidence": result.get('confidence', 'N/A'),
            "best_rerank_score": result.get('best_rerank_score', 0),
            "chunks_used": result.get('chunks_used', 0),
            "chunk_ids": result.get('chunk_ids', []),
            "total_results": len(results),
            "results": results,
            "answer": result.get('answer', ''),
            "retrieval_mode": result.get('retrieval_mode', 'dense'),
            "duration": f"{duration:.2f}s",
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/answer")
async def search_answer(request: AnswerRequest):
    """
    Generate LLM-enhanced answer from textbook content.
    Priority chain: Gemini → Together.ai → OpenRouter.
    Returns structured answer with source chunks for citation.
    """
    start_time = time.time()

    # Resolve textbook ID
    textbook_id = resolve_textbook_id(request.textbook)

    if textbook_id not in loaded_pipelines:
        raise HTTPException(
            status_code=404,
            detail=f"Textbook '{request.textbook}' not found. Available: {list(loaded_pipelines.keys())}"
        )

    pipeline = loaded_pipelines[textbook_id]

    try:
        # Step 1: Run RAG pipeline to get relevant chunks
        rag_start = time.time()
        result = pipeline.query(request.query, mode='raw')
        rag_duration = time.time() - rag_start

        book_relevance = result.get('book_relevance', 0)
        confidence = result.get('confidence', 'N/A')
        chunk_ids = result.get('chunk_ids', [])
        best_rerank_score = result.get('best_rerank_score', 0)
        rerank_scores = result.get('rerank_scores', {})

        # Step 2: Collect actual chunk texts and build source previews
        chunk_texts = []
        search_results = []
        for chunk_id in chunk_ids:
            for meta in pipeline.metadata:
                if meta['chunk_id'] == chunk_id:
                    chunk_texts.append(meta['text'])
                    search_results.append({
                        'chunk_id': chunk_id,
                        'preview': meta['text'][:600],
                        'word_count': meta.get('word_count', len(meta['text'].split())),
                        'score': rerank_scores.get(chunk_id, 0),
                    })
                    break

        # Step 3: Generate LLM answer
        # Always attempt LLM if we have chunks, regardless of confidence label
        answer = result.get('answer', 'No relevant content found.')
        api_used = 'raw'
        model_used = None

        if HAS_LLM and chunk_texts:
            try:
                llm_start = time.time()
                llm_result = generate_answer(chunk_texts[:8], request.query)
                llm_duration = time.time() - llm_start

                if llm_result.get("answer"):
                    answer = llm_result["answer"]
                    api_used = llm_result["api_used"]
                    model_used = llm_result["model"]
                    logger.info(f"LLM answer via {api_used} ({model_used}) in {llm_duration:.1f}s")
                else:
                    logger.warning(f"LLM generation failed: {llm_result.get('error')}")

            except Exception as e:
                logger.warning(f"LLM generation failed, using RAW: {e}")

        duration = time.time() - start_time
        _request_counts['search/answer'] += 1
        _total_latency['search/answer'] += duration
        return {
            "query": request.query,
            "textbook": textbook_id,
            "textbook_name": loaded_textbooks.get(textbook_id, {}).get('name', textbook_id),
            "answer": answer,
            "mode": "llm" if api_used != 'raw' else "raw",
            "api_used": api_used,
            "model": model_used,
            "book_relevance": book_relevance,
            "confidence": confidence,
            "best_rerank_score": best_rerank_score,
            "chunks_used": len(chunk_texts),
            "chunk_ids": chunk_ids,
            "chunks_processed": len(chunk_texts),
            "search_results": search_results,
            "retrieval_mode": result.get('retrieval_mode', 'dense'),
            "duration": f"{duration:.2f}s",
        }

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
