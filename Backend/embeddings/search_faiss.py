#!/usr/bin/env python3
"""
search_faiss.py - Complete Single-File RAG Pipeline with Junk Filtering & Relevance Gate
A production-ready textbook Q&A system with chunking, FAISS indexing, 
cross-encoder reranking, junk filtering, book relevance check, and dual-mode answer generation (RAW/LLM).

Usage:
    # Index a PDF
    python search_faiss.py index --pdf textbook.pdf --id intro_to_ml --name "Intro to ML"
    
    # Query in RAW mode (direct chunk evidence)
    python search_faiss.py query --id intro_to_ml --question "What is supervised learning?" --mode raw
    
    # Query in LLM mode (rewritten answer)
    python search_faiss.py query --id intro_to_ml --question "What is supervised learning?" --mode llm
"""

import os
import re
import sys
import json
import pickle
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np

# Core dependencies
try:
    import faiss
except ImportError:
    print("ERROR: faiss not installed. Run: pip install faiss-cpu")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
except ImportError:
    print("ERROR: nltk not installed. Run: pip install nltk")
    sys.exit(1)

# Optional: PDF extraction
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Optional: subprocess for external LLM script
import subprocess
HAS_LLM_SCRIPT = Path("llm_answer.py").exists()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Heavy embedding models (in order of preference)
EMBEDDING_MODELS = [
    "BAAI/bge-large-en-v1.5",  # 1024-dim, excellent for academic content
    "sentence-transformers/all-mpnet-base-v2",  # 768-dim fallback
]

# Heavy reranking models (in order of preference)
RERANKER_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Fast and good
    "BAAI/bge-reranker-base",  # Good quality
    "mixedbread-ai/mxbai-rerank-large-v1",  # Best quality but slow
]

# LLM models for answer generation (optional)
LLM_MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",  # Fast and good
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Lightweight
]

# Chunking parameters
CHUNK_SIZE_WORDS = 600
OVERLAP_WORDS = 120
MIN_CHUNK_WORDS = 80  # Minimum words for valid chunk

# Retrieval parameters
TOP_K_FAISS = 20
TOP_K_RERANK = 5
RERANK_THRESHOLD = 0.60  # Increased to filter out TOC
BOOK_RELEVANCE_THRESHOLD = 0.40  # Book-level relevance gate

# Directories
INDICES_DIR = Path("indices")
INDICES_DIR.mkdir(exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sigmoid(x):
    """Apply sigmoid to convert logits to probabilities."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def is_junk_chunk(text: str) -> bool:
    """
    Detect and filter junk chunks (TOC, copyright, index pages, etc.).
    
    Args:
        text: Chunk text to evaluate
        
    Returns:
        True if chunk is junk, False otherwise
    """
    low = text.lower()

    # Obvious markers
    if re.search(r"\b(table of contents|contents|copyright|all rights reserved|library of congress)\b", low):
        return True

    # Many short lines = TOC
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 8:
        short_lines = sum(1 for ln in lines if len(ln.split()) <= 6)
        if short_lines / max(1, len(lines)) > 0.55:
            return True

    # Numeric-heavy (indexes)
    tokens = re.split(r"\s+", text)
    if len(tokens) > 0:
        numeric_ratio = sum(1 for t in tokens if re.fullmatch(r"[0-9,\.\-%]+", t)) / len(tokens)
        if numeric_ratio > 0.25:
            return True

    return False


def clean_text(text: str) -> str:
    """Clean extracted text from PDF artifacts."""
    if not text:
        return ""
    
    # Remove hyphenated line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Remove page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Fix common OCR errors
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    
    # Normalize whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\t', ' ', text)
    
    # Remove stray bullets
    text = re.sub(r'^[•▪▫○●◦■□]+\s*', '', text, flags=re.MULTILINE)
    
    # Filter lines - remove numeric junk and clean
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Remove lines that are mostly numbers/punctuation (page listings)
        if re.match(r"^[\d\s\-\.,]{10,}$", line):
            continue
        lines.append(line)
    
    text = '\n'.join(lines)
    
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using available library."""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path.absolute()}")
    
    text = ""
    
    # Try PyMuPDF first (better quality)
    if HAS_PYMUPDF:
        logger.info("Extracting text using PyMuPDF...")
        try:
            doc = fitz.open(str(pdf_path))
            for page in doc:
                text += page.get_text()
            doc.close()
            logger.info(f"Extracted {len(text)} characters")
            return clean_text(text)
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
    
    # Fallback to PyPDF2
    if HAS_PYPDF2:
        logger.info("Extracting text using PyPDF2...")
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
            logger.info(f"Extracted {len(text)} characters")
            return clean_text(text)
        except Exception as e:
            logger.error(f"PyPDF2 failed: {e}")
    
    raise RuntimeError("No PDF library available. Install: pip install PyMuPDF or pip install PyPDF2")


# =============================================================================
# CHUNKING
# =============================================================================

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    chunk_id: int
    text: str
    word_count: int
    sentence_count: int
    start_idx: int
    end_idx: int


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, 
               overlap: int = OVERLAP_WORDS) -> List[Chunk]:
    """
    Chunk text into overlapping segments with complete sentences.
    
    Args:
        text: Input text
        chunk_size: Target chunk size in words
        overlap: Overlap size in words
        
    Returns:
        List of Chunk objects
    """
    if not text.strip():
        return []
    
    # Sentence tokenization
    sentences = sent_tokenize(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_words = []
    current_sentences = []
    overlap_words = []
    chunk_id = 0
    
    for sent_idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        words = sentence.split()
        
        # Check if adding this sentence exceeds chunk size
        if len(current_words) + len(words) > chunk_size and current_words:
            # Only save if meets minimum word count
            if len(current_words) >= MIN_CHUNK_WORDS:
                chunk_text = ' '.join(current_words)
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    word_count=len(current_words),
                    sentence_count=len(current_sentences),
                    start_idx=sent_idx - len(current_sentences),
                    end_idx=sent_idx - 1
                ))
                chunk_id += 1
            
            # Prepare overlap
            overlap_words = current_words[-overlap:] if len(current_words) > overlap else current_words
            current_words = overlap_words.copy()
            
            # Track sentences in overlap
            overlap_sent_count = 0
            word_count = 0
            for sent in reversed(current_sentences):
                sent_words = len(sent.split())
                if word_count + sent_words <= overlap:
                    word_count += sent_words
                    overlap_sent_count += 1
                else:
                    break
            
            current_sentences = current_sentences[-overlap_sent_count:] if overlap_sent_count > 0 else []
        
        # Add sentence to current chunk
        current_words.extend(words)
        current_sentences.append(sentence)
    
    # Add final chunk if meets minimum
    if current_words and len(current_words) >= MIN_CHUNK_WORDS:
        chunk_text = ' '.join(current_words)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            word_count=len(current_words),
            sentence_count=len(current_sentences),
            start_idx=len(sentences) - len(current_sentences),
            end_idx=len(sentences) - 1
        ))
    
    logger.info(f"Created {len(chunks)} chunks (avg {sum(c.word_count for c in chunks) / max(1, len(chunks)):.0f} words/chunk)")
    
    return chunks


# =============================================================================
# EMBEDDING & INDEXING
# =============================================================================

class EmbeddingIndexer:
    """Handles embedding generation and FAISS indexing."""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize embedder.
        
        Args:
            model_name: Specific model or None to auto-select
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        
        # Try to load embedding model
        if model_name:
            models_to_try = [model_name]
        else:
            models_to_try = EMBEDDING_MODELS
        
        self.model = None
        for model in models_to_try:
            try:
                logger.info(f"Loading embedding model: {model}")
                self.model = SentenceTransformer(model, device=device)
                self.model_name = model
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"✓ Loaded {model} (dim={self.embedding_dim})")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model}: {e}")
        
        if self.model is None:
            raise RuntimeError("Could not load any embedding model")
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype('float32')
    
    def build_index(self, chunks: List[Chunk], textbook_id: str, 
                   textbook_name: str) -> Dict:
        """
        Build FAISS index from chunks.
        
        Returns:
            Dictionary with index, metadata, centroid, and config
        """
        logger.info(f"Encoding {len(chunks)} chunks...")
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.encode(texts, show_progress=True)
        
        # Compute book centroid for relevance gating
        logger.info("Computing book centroid...")
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize
        centroid = centroid.astype('float32')
        logger.info(f"✓ Centroid computed: shape {centroid.shape}")
        
        logger.info(f"Building FAISS index (dim={embeddings.shape[1]})...")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        # Save index
        index_path = INDICES_DIR / f"{textbook_id}_index.faiss"
        faiss.write_index(index, str(index_path))
        logger.info(f"✓ Saved FAISS index: {index_path}")
        
        # Save centroid
        centroid_path = INDICES_DIR / f"{textbook_id}_centroid.npy"
        np.save(centroid_path, centroid)
        logger.info(f"✓ Saved centroid: {centroid_path}")
        
        # Save metadata
        metadata = [asdict(chunk) for chunk in chunks]
        metadata_path = INDICES_DIR / f"{textbook_id}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"✓ Saved metadata: {metadata_path}")
        
        # Save config
        config = {
            'textbook_id': textbook_id,
            'textbook_name': textbook_name,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'total_chunks': len(chunks),
            'chunk_size': CHUNK_SIZE_WORDS,
            'overlap': OVERLAP_WORDS
        }
        config_path = INDICES_DIR / f"{textbook_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✓ Saved config: {config_path}")
        
        return {
            'index': index,
            'metadata': metadata,
            'centroid': centroid,
            'config': config
        }


# =============================================================================
# RERANKING
# =============================================================================

class CrossEncoderReranker:
    """Cross-encoder reranking with heavy models."""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize reranker.
        
        Args:
            model_name: Specific model or None to auto-select
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        
        # Try to load reranker model
        if model_name:
            models_to_try = [model_name]
        else:
            models_to_try = RERANKER_MODELS
        
        self.model = None
        for model in models_to_try:
            try:
                logger.info(f"Loading reranker: {model}")
                self.model = CrossEncoder(model, device=device)
                self.model_name = model
                logger.info(f"✓ Loaded {model}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model}: {e}")
        
        if self.model is None:
            raise RuntimeError("Could not load any reranker model")
    
    def rerank(self, query: str, results: List[Dict], 
               top_k: int = TOP_K_RERANK) -> Tuple[List[Dict], float]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Search query
            results: List of search results with 'text' key
            top_k: Number of top results to return
            
        Returns:
            Tuple of (reranked results with 'rerank_score', best_score)
        """
        if not results:
            return [], 0.0
        
        # Prepare pairs
        pairs = [[query, r['text']] for r in results]
        
        # Score
        logger.info(f"Reranking {len(pairs)} results...")
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Apply sigmoid if scores are logits (MS-MARCO style)
        if np.mean(np.abs(scores)) > 5:  # Likely logits
            scores = sigmoid(scores)
        
        # Add scores to results
        for result, score in zip(results, scores):
            result['rerank_score'] = float(score)
        
        # Sort
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Log top scores
        for i, r in enumerate(results[:5]):
            logger.info(f"  Rank {i+1}: score={r['rerank_score']:.4f}, chunk_id={r['chunk_id']}")
        
        # Get best score
        best_score = results[0]['rerank_score'] if results else 0.0
        
        # Filter by threshold
        filtered = [r for r in results if r['rerank_score'] >= RERANK_THRESHOLD]
        
        if not filtered:
            logger.warning(f"No results above threshold {RERANK_THRESHOLD}")
            return [], best_score
        
        return filtered[:top_k], best_score


# =============================================================================
# ANSWER GENERATION
# =============================================================================

class AnswerGenerator:
    """Generate answers in RAW or LLM mode using external llm_answer.py script."""
    
    def __init__(self, use_llm: bool = False, llm_script_path: str = "llm_answer.py"):
        """
        Initialize answer generator.
        
        Args:
            use_llm: Whether to use LLM for answer generation
            llm_script_path: Path to external LLM script
        """
        self.use_llm = use_llm
        self.llm_script_path = Path(llm_script_path)
        
        if use_llm and not self.llm_script_path.exists():
            logger.warning(f"LLM script not found: {llm_script_path}")
            logger.warning("Falling back to RAW mode.")
            self.use_llm = False
    
    def generate_raw_answer(self, chunks: List[Dict], query: str) -> str:
        """
        Generate RAW answer by concatenating top chunks.
        No LLM, no rewriting - just direct evidence.
        """
        if not chunks:
            return "No relevant content found in the textbook."
        
        # Combine top chunks without modification
        answer_parts = []
        for i, chunk in enumerate(chunks[:3], 1):
            text = chunk['text'].strip()
            answer_parts.append(f"[Evidence {i} - Chunk ID: {chunk['chunk_id']}]\n{text}")
        
        answer = "\n\n".join(answer_parts)
        return answer
    
    def generate_llm_answer(self, chunks: List[Dict], query: str) -> str:
        """
        Generate LLM-enhanced answer using external llm_answer.py script.
        Calls Together.ai or OpenRouter API via subprocess.
        """
        if not chunks:
            return "No relevant content found in the textbook."
        
        if not self.use_llm:
            return self.generate_raw_answer(chunks, query)
        
        try:
            # Prepare chunk texts (top 5 chunks)
            chunk_texts = [c['text'] for c in chunks[:5]]
            
            # Build command
            cmd = ["python", str(self.llm_script_path), query] + chunk_texts
            
            logger.info("Calling external LLM script...")
            
            # Call external script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"LLM script failed with return code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                logger.info("Falling back to RAW mode")
                return self.generate_raw_answer(chunks, query)
            
            # Parse JSON response
            try:
                response_data = json.loads(result.stdout)
                
                if "error" in response_data:
                    logger.error(f"LLM API error: {response_data.get('message', 'Unknown error')}")
                    logger.info("Falling back to RAW mode")
                    return self.generate_raw_answer(chunks, query)
                
                answer = response_data.get('answer', '').strip()
                api_used = response_data.get('api_used', 'unknown')
                
                if answer:
                    logger.info(f"✓ LLM answer generated via {api_used}")
                    return answer
                else:
                    logger.warning("LLM returned empty answer")
                    return self.generate_raw_answer(chunks, query)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response: {e}")
                logger.error(f"stdout: {result.stdout[:500]}")
                logger.info("Falling back to RAW mode")
                return self.generate_raw_answer(chunks, query)
        
        except subprocess.TimeoutExpired:
            logger.error("LLM script timeout (120s)")
            logger.info("Falling back to RAW mode")
            return self.generate_raw_answer(chunks, query)
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            logger.info("Falling back to RAW mode")
            return self.generate_raw_answer(chunks, query)
    
    def generate(self, chunks: List[Dict], query: str, use_llm: bool) -> str:
        """Generate answer based on mode (RAW or LLM)."""
        if use_llm:
            return self.generate_llm_answer(chunks, query)
        else:
            return self.generate_raw_answer(chunks, query)


# =============================================================================
# SEARCH PIPELINE
# =============================================================================

class SearchPipeline:
    """Complete search pipeline with FAISS + reranking + junk filtering."""
    
    def __init__(self, textbook_id: str, device: str = None):
        """
        Initialize search pipeline.
        
        Args:
            textbook_id: Textbook identifier
            device: 'cpu' or 'cuda'
        """
        self.textbook_id = textbook_id
        self.device = device
        
        # Load config
        config_path = INDICES_DIR / f"{textbook_id}_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Index not found for textbook: {textbook_id}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"Loading index for: {self.config['textbook_name']}")
        
        # Load FAISS index
        index_path = INDICES_DIR / f"{textbook_id}_index.faiss"
        self.index = faiss.read_index(str(index_path))
        logger.info(f"✓ Loaded FAISS index: {self.index.ntotal} vectors")
        
        # Load centroid (or compute if missing)
        centroid_path = INDICES_DIR / f"{textbook_id}_centroid.npy"
        if centroid_path.exists():
            self.centroid = np.load(centroid_path)
            logger.info(f"✓ Loaded book centroid: shape {self.centroid.shape}")
        else:
            logger.warning(f"Centroid not found. Computing from index...")
            # Extract all vectors from FAISS index
            all_vectors = np.zeros((self.index.ntotal, self.index.d), dtype='float32')
            for i in range(self.index.ntotal):
                all_vectors[i] = self.index.reconstruct(i)
            # Compute centroid
            self.centroid = np.mean(all_vectors, axis=0, keepdims=True)
            self.centroid = self.centroid / np.linalg.norm(self.centroid)
            # Save for future use
            np.save(centroid_path, self.centroid)
            logger.info(f"✓ Computed and saved centroid: shape {self.centroid.shape}")
        
        # Load metadata
        metadata_path = INDICES_DIR / f"{textbook_id}_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        logger.info(f"✓ Loaded {len(self.metadata)} chunks")
        
        # Initialize components
        self.embedder = EmbeddingIndexer(
            model_name=self.config['model_name'],
            device=device
        )
        
        self.reranker = CrossEncoderReranker(device=device)
    
    def check_book_relevance(self, query_embedding: np.ndarray) -> float:
        """
        Check if query is relevant to the book.
        
        Args:
            query_embedding: Query embedding (1 x D)
            
        Returns:
            Cosine similarity score with book centroid
        """
        similarity = np.dot(query_embedding, self.centroid.T)[0, 0]
        return float(similarity)
    
    def search(self, query: str, top_k: int = TOP_K_FAISS) -> Tuple[List[Dict], np.ndarray]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results from FAISS
            
        Returns:
            Tuple of (results list, query_embedding)
        """
        if not query.strip():
            return [], None
        
        # Encode query
        logger.info(f"Searching: '{query}'")
        query_embedding = self.embedder.encode([query], show_progress=False)
        
        # Search FAISS
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                chunk = self.metadata[idx]
                results.append({
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'word_count': chunk['word_count'],
                    'faiss_score': float(score)
                })
        
        logger.info(f"✓ Found {len(results)} results from FAISS")
        
        return results, query_embedding
    
    def query(self, question: str, mode: str = 'raw') -> Dict:
        """
        Complete query pipeline with relevance gating.
        
        Args:
            question: User question
            mode: 'raw' or 'llm'
            
        Returns:
            Dictionary with answer and metadata
        """
        # Search
        results, query_embedding = self.search(question, top_k=TOP_K_FAISS)
        
        # Check book-level relevance FIRST
        book_relevance = self.check_book_relevance(query_embedding)
        logger.info(f"Book relevance score: {book_relevance:.4f}")
        
        if book_relevance < BOOK_RELEVANCE_THRESHOLD:
            logger.warning(f"Book relevance {book_relevance:.4f} < threshold {BOOK_RELEVANCE_THRESHOLD}")
            logger.warning("Query is not relevant to this textbook.")
            return {
                'answer': "The textbook does not contain relevant information related to your question.",
                'textbook_name': self.config['textbook_name'],
                'question': question,
                'mode': mode,
                'book_relevance': book_relevance,
                'chunks_used': [],
                'chunk_ids': [],
                'best_rerank_score': 0.0,
                'confidence': 'N/A'
            }
        
        if not results:
            return {
                'answer': "No relevant content found in the textbook.",
                'textbook_name': self.config['textbook_name'],
                'question': question,
                'mode': mode,
                'book_relevance': book_relevance,
                'chunks_used': [],
                'chunk_ids': [],
                'best_rerank_score': 0.0,
                'confidence': 'Low'
            }
        
        # Filter junk chunks
        logger.info("Filtering junk chunks...")
        filtered_results = []
        for r in results:
            if not is_junk_chunk(r['text']):
                filtered_results.append(r)
            else:
                logger.info(f"  Filtered junk chunk id={r['chunk_id']}")
        
        if not filtered_results:
            logger.warning("All retrieved chunks were filtered as junk; using original set.")
            filtered_results = results
        else:
            logger.info(f"✓ Kept {len(filtered_results)} clean chunks (filtered {len(results) - len(filtered_results)} junk)")
        
        # Rerank
        reranked, best_rerank_score = self.reranker.rerank(question, filtered_results, top_k=TOP_K_RERANK)
        
        if not reranked:
            return {
                'answer': "No relevant content found in the textbook.",
                'textbook_name': self.config['textbook_name'],
                'question': question,
                'mode': mode,
                'book_relevance': book_relevance,
                'chunks_used': [],
                'chunk_ids': [],
                'best_rerank_score': best_rerank_score,
                'confidence': 'Low'
            }
        
        # Classify confidence
        if best_rerank_score >= 0.75:
            confidence = 'High'
        elif best_rerank_score >= 0.65:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        # Generate answer
        use_llm = (mode.lower() == 'llm')
        generator = AnswerGenerator(use_llm=use_llm)
        answer = generator.generate(reranked, question, use_llm=use_llm)
        
        # Extract chunk IDs
        chunk_ids = [r['chunk_id'] for r in reranked]
        
        return {
            'answer': answer,
            'textbook_name': self.config['textbook_name'],
            'question': question,
            'mode': mode,
            'book_relevance': book_relevance,
            'chunks_used': len(reranked),
            'chunk_ids': chunk_ids,
            'best_rerank_score': best_rerank_score,
            'confidence': confidence
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def cmd_index(args):
    """Index a PDF textbook."""
    logger.info("="*70)
    logger.info("INDEXING PIPELINE")
    logger.info("="*70)
    
    # Extract text
    if args.pdf:
        text = extract_text_from_pdf(args.pdf)
    elif args.text:
        with open(args.text, 'r', encoding='utf-8') as f:
            text = f.read()
        text = clean_text(text)
    else:
        logger.error("Must provide --pdf or --text")
        sys.exit(1)
    
    if not text.strip():
        logger.error("Extracted text is empty")
        sys.exit(1)
    
    logger.info(f"Extracted {len(text)} characters")
    
    # Chunk
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=OVERLAP_WORDS)
    
    if not chunks:
        logger.error("No chunks created")
        sys.exit(1)
    
    # Build index
    indexer = EmbeddingIndexer(device=args.device)
    indexer.build_index(
        chunks=chunks,
        textbook_id=args.id,
        textbook_name=args.name or args.id
    )
    
    logger.info("="*70)
    logger.info("✓ INDEXING COMPLETE")
    logger.info("="*70)


def cmd_query(args):
    """Query a textbook."""
    logger.info("="*70)
    logger.info("QUERY PIPELINE")
    logger.info("="*70)
    
    # Initialize pipeline
    pipeline = SearchPipeline(textbook_id=args.id, device=args.device)
    
    # Query
    result = pipeline.query(args.question, mode=args.mode)
    
    # Display
    print("\n" + "="*70)
    print(f"TEXTBOOK: {result['textbook_name']}")
    print(f"QUESTION: {result['question']}")
    print(f"MODE: {result['mode'].upper()}")
    print("="*70)
    print(f"\nBOOK RELEVANCE: {result['book_relevance']:.4f}")
    print(f"CONFIDENCE: {result['confidence']}")
    print(f"CHUNKS USED: {result['chunks_used']}")
    
    if result['chunk_ids']:
        # IMPORTANT: Output as proper JSON array string
        import json as json_module
        print(f"CHUNK_IDS: {json_module.dumps(result['chunk_ids'])}")
    
    if result['best_rerank_score'] > 0:
        print(f"BEST RERANK SCORE: {result['best_rerank_score']:.4f}")
    
    print("\nANSWER:")
    print("-"*70)
    print(result['answer'])
    print("-"*70)


def main():
    parser = argparse.ArgumentParser(
        description="Single-File RAG Pipeline for Textbook Q&A with Junk Filtering & Relevance Gate",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index a textbook')
    index_parser.add_argument('--pdf', help='Path to PDF file')
    index_parser.add_argument('--text', help='Path to text file')
    index_parser.add_argument('--id', required=True, help='Textbook ID')
    index_parser.add_argument('--name', help='Textbook name')
    index_parser.add_argument('--device', default=None, help='Device: cpu or cuda')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query a textbook')
    query_parser.add_argument('--id', required=True, help='Textbook ID')
    query_parser.add_argument('--question', required=True, help='Question to ask')
    query_parser.add_argument('--mode', default='raw', choices=['raw', 'llm'], 
                             help='Answer mode: raw (direct chunks) or llm (rewritten)')
    query_parser.add_argument('--device', default=None, help='Device: cpu or cuda')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'index':
        cmd_index(args)
    elif args.command == 'query':
        cmd_query(args)


if __name__ == "__main__":
    main()