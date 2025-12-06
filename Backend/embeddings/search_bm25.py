#!/usr/bin/env python3
"""
Enhanced BM25 Search with Advanced Optimizations

Key Improvements:
1. Better tokenization with stemming and stopword removal
2. Query expansion for better recall
3. Phrase matching bonus
4. Title/heading boost
5. Configurable BM25 parameters
6. Result re-ranking with semantic similarity
"""

import argparse
import pickle
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import re
from collections import Counter
import math

import numpy as np


class EnhancedBM25Ranker:
    """Enhanced BM25 with multiple improvements for better accuracy."""
    
    def __init__(
        self, 
        k1: float = 1.2,  # Reduced from 1.5 for better term frequency saturation
        b: float = 0.75,
        use_stemming: bool = True,
        remove_stopwords: bool = True,
        boost_phrases: bool = True,
        boost_titles: bool = True
    ):
        """
        Initialize Enhanced BM25 ranker.
        
        Args:
            k1: Term frequency saturation (1.0-2.0, lower = faster saturation)
            b: Length normalization (0.0-1.0, higher = more normalization)
            use_stemming: Apply Porter stemming
            remove_stopwords: Remove common stopwords
            boost_phrases: Give bonus to exact phrase matches
            boost_titles: Boost chunks with query terms in titles
        """
        self.k1 = k1
        self.b = b
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.boost_phrases = boost_phrases
        self.boost_titles = boost_titles
        
        self.documents = []
        self.original_docs = []  # Keep original for phrase matching
        self.tokenized_docs = []
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_count = 0
        self.idf_scores = {}
        
        # Stopwords (common English words with minimal semantic value)
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
            'what', 'when', 'where', 'who', 'which', 'why', 'how'
        }
        
    def simple_stem(self, word: str) -> str:
        """
        Simple stemming (removes common suffixes).
        For production, use nltk.stem.PorterStemmer or similar.
        """
        if not self.use_stemming:
            return word
            
        # Remove common suffixes
        suffixes = ['ing', 'ed', 'es', 's', 'ly', 'tion', 'ation', 'ness', 'ment']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def tokenize(self, text: str, remove_stops: bool = None) -> List[str]:
        """
        Enhanced tokenization with optional stemming and stopword removal.
        
        Args:
            text: Input text
            remove_stops: Override class setting for stopword removal
            
        Returns:
            List of processed tokens
        """
        if remove_stops is None:
            remove_stops = self.remove_stopwords
        
        # Extract words (alphanumeric + hyphens for terms like "cross-validation")
        tokens = re.findall(r'\b[\w-]+\b', text.lower())
        
        # Remove stopwords if enabled
        if remove_stops:
            tokens = [t for t in tokens if t not in self.stopwords and len(t) > 2]
        
        # Apply stemming
        tokens = [self.simple_stem(t) for t in tokens]
        
        return tokens
    
    def fit(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """
        Fit BM25 model on document corpus.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for title boosting
        """
        self.documents = documents
        self.original_docs = documents  # Keep for phrase matching
        self.doc_count = len(documents)
        self.metadata = metadata or [{}] * len(documents)
        
        # Tokenize all documents
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]
        
        # Calculate document lengths
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0
        
        # Calculate IDF scores
        self._calculate_idf()
    
    def _calculate_idf(self):
        """Calculate IDF with improved formula (prevents negative IDF)."""
        df = Counter()
        
        for doc_tokens in self.tokenized_docs:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                df[token] += 1
        
        # Robertson's IDF formula (always positive)
        for term, freq in df.items():
            idf = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf_scores[term] = max(idf, 0.0)  # Ensure non-negative
    
    def score_document(
        self, 
        query_tokens: List[str], 
        doc_idx: int,
        original_query: str = ""
    ) -> float:
        """
        Calculate enhanced BM25 score with boosting.
        
        Args:
            query_tokens: Tokenized query
            doc_idx: Document index
            original_query: Original query for phrase matching
            
        Returns:
            Enhanced BM25 score
        """
        base_score = 0.0
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        # Count term frequencies
        doc_term_freq = Counter(doc_tokens)
        
        # Calculate base BM25 score
        for token in query_tokens:
            if token not in doc_term_freq:
                continue
            
            idf = self.idf_scores.get(token, 0)
            tf = doc_term_freq[token]
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            
            base_score += idf * (numerator / denominator)
        
        # Apply boosting factors
        boost_multiplier = 1.0
        
        # Phrase matching bonus (exact phrase in document)
        if self.boost_phrases and original_query:
            if original_query.lower() in self.original_docs[doc_idx].lower():
                boost_multiplier *= 1.5  # 50% boost for exact phrase match
        
        # Title/heading boost
        if self.boost_titles and self.metadata:
            meta = self.metadata[doc_idx]
            title = meta.get('title', '') or meta.get('chunk_id', '')
            
            # Check if query terms appear in title
            title_tokens = self.tokenize(title)
            query_in_title = sum(1 for qt in query_tokens if qt in title_tokens)
            
            if query_in_title > 0:
                # Boost based on proportion of query terms in title
                title_boost = 1.0 + (0.3 * query_in_title / len(query_tokens))
                boost_multiplier *= title_boost
        
        return base_score * boost_multiplier
    
    def expand_query(self, query_tokens: List[str]) -> List[str]:
        """
        Simple query expansion using token variations.
        
        For production, consider using WordNet synonyms or word embeddings.
        """
        expanded = query_tokens.copy()
        
        # Add common variations (you can expand this)
        variations = {
            'neural': ['network', 'neuron'],
            'machine': ['learn', 'algorithm'],
            'deep': ['neural', 'network'],
            'train': ['fit', 'model'],
            'test': ['validation', 'evaluate'],
            'accuracy': ['performance', 'metric'],
        }
        
        for token in query_tokens:
            if token in variations:
                expanded.extend(variations[token])
        
        return list(set(expanded))  # Remove duplicates
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        expand_query: bool = True,
        diversity: bool = False
    ) -> List[Tuple[int, float]]:
        """
        Enhanced search with query expansion and diversity.
        
        Args:
            query: Search query
            top_k: Number of results
            expand_query: Enable query expansion
            diversity: Enable result diversification (MMR-style)
            
        Returns:
            List of (doc_index, score) tuples
        """
        query_tokens = self.tokenize(query)
        
        # Query expansion
        if expand_query:
            query_tokens = self.expand_query(query_tokens)
        
        # Score all documents
        scores = []
        for doc_idx in range(self.doc_count):
            score = self.score_document(query_tokens, doc_idx, query)
            scores.append((doc_idx, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Optional: Diversity (simple MMR-style)
        if diversity and len(scores) > top_k:
            diverse_results = self._diversify_results(scores, top_k)
            return diverse_results
        
        return scores[:top_k]
    
    def _diversify_results(
        self, 
        scored_results: List[Tuple[int, float]], 
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Diversify results to avoid redundancy (simple version).
        Returns results that are both relevant and diverse.
        """
        if len(scored_results) <= top_k:
            return scored_results[:top_k]
        
        selected = [scored_results[0]]  # Start with top result
        candidates = scored_results[1:top_k * 3]  # Consider top 3*k candidates
        
        lambda_param = 0.7  # Balance relevance vs diversity
        
        while len(selected) < top_k and candidates:
            best_score = -float('inf')
            best_idx = 0
            
            for i, (doc_idx, rel_score) in enumerate(candidates):
                # Calculate diversity (simple: based on text overlap)
                div_score = self._calculate_diversity(doc_idx, [s[0] for s in selected])
                
                # MMR-style scoring
                mmr_score = lambda_param * rel_score + (1 - lambda_param) * div_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(candidates.pop(best_idx))
        
        return selected
    
    def _calculate_diversity(self, doc_idx: int, selected_indices: List[int]) -> float:
        """Calculate diversity score (inverse of similarity to selected docs)."""
        if not selected_indices:
            return 1.0
        
        doc_tokens = set(self.tokenized_docs[doc_idx])
        
        # Average dissimilarity to already selected documents
        similarities = []
        for sel_idx in selected_indices:
            sel_tokens = set(self.tokenized_docs[sel_idx])
            
            # Jaccard similarity
            intersection = len(doc_tokens & sel_tokens)
            union = len(doc_tokens | sel_tokens)
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity  # Convert to diversity


class TextbookBM25Searcher:
    """Enhanced BM25-based search for textbook collections."""
    
    def __init__(
        self,
        textbook_id: str,
        json_mode: bool = False,
        indices_dir: str = "indices",
        k1: float = 1.2,
        b: float = 0.75,
        use_enhancements: bool = True
    ):
        """Initialize enhanced BM25 searcher."""
        self.textbook_id = textbook_id
        self.json_mode = json_mode
        self.indices_dir = Path(indices_dir)
        self.use_enhancements = use_enhancements
        
        # File paths
        self.bm25_index_path = self.indices_dir / f"{textbook_id}_bm25_index.pkl"
        self.metadata_path = self.indices_dir / f"{textbook_id}_metadata.pkl"
        self.config_path = self.indices_dir / f"{textbook_id}_config.json"
        
        self.bm25_ranker = None
        self.metadata = None
        self.config = None
        
        # Load components
        self._load_config()
        self._load_metadata()
        
        # Load or create enhanced index
        if use_enhancements:
            self._create_enhanced_index(k1, b)
        else:
            self._load_bm25_index()
        
        if not self.json_mode:
            textbook_name = self.config.get('textbook_name', textbook_id)
            enhancements = " (Enhanced)" if use_enhancements else ""
            print(f"SUCCESS: BM25 Searcher{enhancements} initialized for '{textbook_name}' with {len(self.metadata)} chunks")
    
    def _log(self, message: str):
        """Log message only if not in JSON mode."""
        if not self.json_mode:
            print(message)
    
    def _load_config(self):
        """Load textbook configuration."""
        try:
            if not self.config_path.exists():
                error_msg = f"Config file not found: {self.config_path}"
                if self.json_mode:
                    print(json.dumps({"error": error_msg}))
                else:
                    print(f"ERROR: {error_msg}")
                sys.exit(1)
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
        except Exception as e:
            error_msg = f"Loading config failed: {str(e)}"
            if self.json_mode:
                print(json.dumps({"error": error_msg}))
            else:
                print(f"ERROR: {error_msg}")
            sys.exit(1)
    
    def _load_metadata(self):
        """Load metadata mapping."""
        try:
            if not self.metadata_path.exists():
                error_msg = f"Metadata file not found: {self.metadata_path}"
                if self.json_mode:
                    print(json.dumps({"error": error_msg}))
                else:
                    print(f"ERROR: {error_msg}")
                sys.exit(1)
            
            with open(self.metadata_path, 'rb') as file:
                self.metadata = pickle.load(file)
            
        except Exception as e:
            error_msg = f"Loading metadata failed: {str(e)}"
            if self.json_mode:
                print(json.dumps({"error": error_msg}))
            else:
                print(f"ERROR: {error_msg}")
            sys.exit(1)
    
    def _create_enhanced_index(self, k1: float, b: float):
        """Create enhanced BM25 index with optimizations."""
        self._log("Creating enhanced BM25 index...")
        
        # Extract documents from metadata
        documents = [meta.get('text', '') for meta in self.metadata]
        
        # Create enhanced ranker
        self.bm25_ranker = EnhancedBM25Ranker(
            k1=k1,
            b=b,
            use_stemming=True,
            remove_stopwords=True,
            boost_phrases=True,
            boost_titles=True
        )
        
        # Fit on documents
        self.bm25_ranker.fit(documents, self.metadata)
        
        self._log("Enhanced BM25 index created successfully")
    
    def _load_bm25_index(self):
        """Load existing BM25 index."""
        try:
            if not self.bm25_index_path.exists():
                error_msg = f"BM25 index not found: {self.bm25_index_path}"
                if self.json_mode:
                    print(json.dumps({
                        "error": error_msg,
                        "hint": "Run bm25_indexer.py to create BM25 index"
                    }))
                else:
                    print(f"ERROR: {error_msg}")
                sys.exit(1)
            
            with open(self.bm25_index_path, 'rb') as f:
                self.bm25_ranker = pickle.load(f)
            
        except Exception as e:
            error_msg = f"Loading BM25 index failed: {str(e)}"
            if self.json_mode:
                print(json.dumps({"error": error_msg}))
            else:
                print(f"ERROR: {error_msg}")
            sys.exit(1)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Enhanced search with query expansion and boosting.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (score, metadata) tuples
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Get enhanced BM25 scores
        results = self.bm25_ranker.search(
            query.strip(), 
            min(top_k, len(self.metadata)),
            expand_query=self.use_enhancements,
            diversity=False  # Can enable for more diverse results
        )
        
        # Format results with metadata
        formatted_results = []
        for doc_idx, score in results:
            if doc_idx < len(self.metadata):
                metadata = self.metadata[doc_idx].copy()
                metadata['textbook_id'] = self.textbook_id
                metadata['textbook_name'] = self.config.get('textbook_name', self.textbook_id)
                formatted_results.append((float(score), metadata))
        
        return formatted_results
    
    def format_results_json(
        self,
        results: List[Tuple[float, Dict[str, Any]]],
        query: str
    ) -> Dict[str, Any]:
        """Format results as JSON."""
        if not results:
            return {
                "query": query,
                "textbook": {
                    "id": self.textbook_id,
                    "name": self.config.get('textbook_name', self.textbook_id)
                },
                "search_method": "Enhanced BM25",
                "total_results": 0,
                "results": [],
                "message": "No relevant results found"
            }
        
        formatted_results = []
        for rank, (score, metadata) in enumerate(results, 1):
            result_item = {
                "rank": rank,
                "score": round(score, 4),
                "chunk_id": metadata.get('chunk_id', 'Unknown'),
                "content": metadata.get('text', 'No text available'),
                "word_count": metadata.get('word_count', 0),
                "textbook_id": metadata.get('textbook_id', self.textbook_id),
                "textbook_name": metadata.get('textbook_name', self.textbook_id)
            }
            formatted_results.append(result_item)
        
        return {
            "query": query,
            "textbook": {
                "id": self.textbook_id,
                "name": self.config.get('textbook_name', self.textbook_id)
            },
            "search_method": "Enhanced BM25",
            "enhancements": "stemming, stopwords, phrase_boost, title_boost, query_expansion",
            "total_results": len(results),
            "results": formatted_results
        }
    
    def format_results(
        self,
        results: List[Tuple[float, Dict[str, Any]]],
        query: str,
        show_scores: bool = False
    ) -> str:
        """Format results for display."""
        if not results:
            return "No results found."
        
        textbook_name = self.config.get('textbook_name', self.textbook_id)
        
        output = []
        output.append("=" * 60)
        output.append(f"TEXTBOOK: {textbook_name}")
        output.append(f"SEARCH METHOD: Enhanced BM25")
        output.append(f"QUERY: \"{query}\"")
        output.append(f"FOUND: {len(results)} relevant chunks")
        output.append("=" * 60)
        
        for rank, (score, metadata) in enumerate(results, 1):
            output.append(f"\nRANK {rank}")
            
            if show_scores:
                output.append(f"BM25 SCORE: {score:.4f}")
            
            chunk_id = metadata.get('chunk_id', 'Unknown')
            word_count = metadata.get('word_count', 'Unknown')
            output.append(f"ID: {chunk_id} | WORDS: {word_count}")
            
            text = metadata.get('text', 'No text available')
            if len(text) > 500:
                text = text[:500] + "..."
            
            output.append("TEXT:")
            output.append(f"   {text}")
            
            if rank < len(results):
                output.append("-" * 40)
        
        return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced BM25 keyword-based search for textbook chunks"
    )
    
    parser.add_argument('--textbook', '-t', required=True,
                       help='Textbook ID (e.g., intro_to_ml)')
    parser.add_argument('--query', '-q', help='Search query')
    parser.add_argument('--top_k', '-k', type=int, default=5,
                       help='Number of results (default: 5)')
    parser.add_argument('--k1', type=float, default=1.2,
                       help='BM25 k1 parameter (default: 1.2)')
    parser.add_argument('--b', type=float, default=0.75,
                       help='BM25 b parameter (default: 0.75)')
    parser.add_argument('--no_enhancements', action='store_true',
                       help='Disable enhancements (use basic BM25)')
    parser.add_argument('--show_scores', action='store_true',
                       help='Show BM25 scores')
    parser.add_argument('--json', action='store_true',
                       help='JSON output')
    parser.add_argument('--indices_dir', default='indices',
                       help='Indices directory')
    
    args = parser.parse_args()
    
    try:
        searcher = TextbookBM25Searcher(
            textbook_id=args.textbook,
            json_mode=args.json,
            indices_dir=args.indices_dir,
            k1=args.k1,
            b=args.b,
            use_enhancements=not args.no_enhancements
        )
        
        if args.query:
            results = searcher.search(args.query, args.top_k)
            
            if args.json:
                json_results = searcher.format_results_json(results, args.query)
                print(json.dumps(json_results, indent=2, ensure_ascii=False))
            else:
                formatted = searcher.format_results(results, args.query, args.show_scores)
                print(formatted)
        else:
            if args.json:
                print(json.dumps({"error": "No query provided"}))
                return 1
            
            query = input("QUESTION: ").strip()
            if query:
                results = searcher.search(query, args.top_k)
                formatted = searcher.format_results(results, query, args.show_scores)
                print(formatted)
        
        return 0
        
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"ERROR: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())