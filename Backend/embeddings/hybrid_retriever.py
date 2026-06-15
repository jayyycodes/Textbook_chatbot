#!/usr/bin/env python3
"""
hybrid_retriever.py - Hybrid BM25 + FAISS Retrieval with Reciprocal Rank Fusion

Interview talking point:
    Dense retrieval (FAISS) excels at semantic similarity.
    Sparse retrieval (BM25) excels at exact keyword matching.
    RRF fusion combines both without needing score normalization.
    This consistently outperforms either method alone on academic content.
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("rank-bm25 not installed. Run: pip install rank-bm25")


# =============================================================================
# RECIPROCAL RANK FUSION
# =============================================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) for each list where item appears.
    k=60 is the standard value from the original RRF paper (Cormack 2009).

    Args:
        ranked_lists: List of ranked index lists (most relevant first)
        k: RRF constant (higher k = less penalty for lower ranks)

    Returns:
        List of (idx, rrf_score) sorted by score descending
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# =============================================================================
# BM25 INDEX BUILDER
# =============================================================================

def build_bm25_index(metadata: List[dict], save_path: Path) -> Optional["BM25Okapi"]:
    """
    Build and persist a BM25 index from chunk metadata.

    Args:
        metadata: List of chunk dicts with 'text' field
        save_path: Path to save the pickled BM25 index

    Returns:
        BM25Okapi instance or None if rank_bm25 not available
    """
    if not HAS_BM25:
        return None

    logger.info(f"Building BM25 index for {len(metadata)} chunks...")
    tokenized = [doc['text'].lower().split() for doc in metadata]
    bm25 = BM25Okapi(tokenized)

    with open(save_path, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"✓ BM25 index saved: {save_path}")
    return bm25


def load_bm25_index(path: Path) -> Optional["BM25Okapi"]:
    """Load a persisted BM25 index."""
    if not path.exists() or not HAS_BM25:
        return None
    try:
        with open(path, 'rb') as f:
            bm25 = pickle.load(f)
        logger.info(f"✓ Loaded BM25 index: {path}")
        return bm25
    except Exception as e:
        logger.warning(f"Failed to load BM25 index: {e}")
        return None


# =============================================================================
# HYBRID RETRIEVER
# =============================================================================

class HybridRetriever:
    """
    Hybrid retriever combining FAISS dense search and BM25 sparse search via RRF.

    Architecture:
        Query → FAISS top-N (semantic) + BM25 top-N (keyword)
              → RRF Fusion
              → top-K candidates for reranking
    """

    def __init__(self, faiss_index, metadata: List[dict], embedder, bm25=None):
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.embedder = embedder
        self.bm25 = bm25
        self.using_hybrid = bm25 is not None and HAS_BM25
        mode = "Hybrid (FAISS+BM25+RRF)" if self.using_hybrid else "Dense-only (FAISS)"
        logger.info(f"HybridRetriever initialized — mode: {mode}")

    def dense_search(self, query_embedding: np.ndarray, top_k: int) -> List[int]:
        """Return ranked list of metadata indices from FAISS."""
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        return [int(idx) for idx in indices[0] if idx < len(self.metadata)]

    def sparse_search(self, query: str, top_k: int) -> List[int]:
        """Return ranked list of metadata indices from BM25."""
        if not self.bm25:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in ranked if scores[i] > 0]

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Retrieve top-k candidates using hybrid RRF fusion.

        Returns list of result dicts with chunk data.
        """
        dense_ranked = self.dense_search(query_embedding, top_k)

        if self.using_hybrid:
            sparse_ranked = self.sparse_search(query, top_k)
            fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked])
        else:
            # Fall back to dense-only
            fused = [(idx, 1.0 / (60 + rank + 1)) for rank, idx in enumerate(dense_ranked)]

        results = []
        seen = set()
        for idx, rrf_score in fused[:top_k]:
            if idx in seen or idx >= len(self.metadata):
                continue
            seen.add(idx)
            chunk = self.metadata[idx]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'word_count': chunk.get('word_count', len(chunk['text'].split())),
                'faiss_score': rrf_score,  # Store RRF score in faiss_score field
                'retrieval_mode': 'hybrid' if self.using_hybrid else 'dense',
            })

        logger.info(
            f"✓ Retrieved {len(results)} candidates "
            f"({'hybrid RRF' if self.using_hybrid else 'dense only'})"
        )
        return results
