#!/usr/bin/env python3
"""
BM25 Indexer for Textbook Chatbot

Creates BM25 indices from textbook chunks for keyword-based search.

Usage:
    python bm25_indexer.py --textbook intro_to_ml
    python bm25_indexer.py --textbook computer_networks --k1 1.2 --b 0.8
"""

import argparse
import pickle
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import re
from collections import Counter
import math

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Progress bar will not be shown.")
    tqdm = lambda x, **kwargs: x


class BM25Ranker:
    """BM25 ranking implementation."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.tokenized_docs = []
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_count = 0
        self.idf_scores = {}
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def fit(self, documents: List[str]):
        """Fit BM25 model on document corpus."""
        print(f"🔄 Tokenizing {len(documents)} documents...")
        self.documents = documents
        self.doc_count = len(documents)
        
        # Tokenize with progress bar
        self.tokenized_docs = []
        for doc in tqdm(documents, desc="Tokenizing"):
            self.tokenized_docs.append(self.tokenize(doc))
        
        # Calculate document lengths
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0
        
        print(f"📊 Average document length: {self.avgdl:.2f} tokens")
        
        # Calculate IDF scores
        print("🔄 Calculating IDF scores...")
        self._calculate_idf()
        print(f"✅ Vocabulary size: {len(self.idf_scores)} unique terms")
    
    def _calculate_idf(self):
        """Calculate Inverse Document Frequency."""
        df = Counter()
        
        for doc_tokens in self.tokenized_docs:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                df[token] += 1
        
        for term, freq in df.items():
            idf = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
            self.idf_scores[term] = idf
    
    def score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        score = 0.0
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        doc_term_freq = Counter(doc_tokens)
        
        for token in query_tokens:
            if token not in doc_term_freq:
                continue
            
            idf = self.idf_scores.get(token, 0)
            tf = doc_term_freq[token]
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """Search documents using BM25."""
        query_tokens = self.tokenize(query)
        
        scores = []
        for doc_idx in range(self.doc_count):
            score = self.score_document(query_tokens, doc_idx)
            scores.append((doc_idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """Load metadata from pickle file."""
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✅ Loaded {len(metadata)} chunks from metadata")
        return metadata
    except Exception as e:
        print(f"❌ Error loading metadata: {str(e)}")
        sys.exit(1)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load textbook configuration."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Loaded config for: {config.get('textbook_name', 'Unknown')}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {str(e)}")
        sys.exit(1)


def create_bm25_index(
    textbook_id: str,
    indices_dir: str = "indices",
    k1: float = 1.5,
    b: float = 0.75
) -> None:
    """Create BM25 index for a textbook."""
    
    indices_path = Path(indices_dir)
    
    # Paths
    metadata_path = indices_path / f"{textbook_id}_metadata.pkl"
    config_path = indices_path / f"{textbook_id}_config.json"
    output_path = indices_path / f"{textbook_id}_bm25_index.pkl"
    
    print("=" * 60)
    print("🚀 BM25 INDEX CREATION")
    print("=" * 60)
    print(f"Textbook ID: {textbook_id}")
    print(f"BM25 Parameters: k1={k1}, b={b}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Check if files exist
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        print("💡 Run embedding_indexer.py first to create metadata")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    # Load data
    config = load_config(config_path)
    metadata = load_metadata(metadata_path)
    
    # Extract texts
    print("\n🔄 Extracting texts from chunks...")
    documents = [chunk.get('text', '') for chunk in metadata]
    
    # Validate documents
    valid_docs = [doc for doc in documents if doc.strip()]
    if len(valid_docs) < len(documents):
        print(f"⚠️  Filtered out {len(documents) - len(valid_docs)} empty chunks")
    
    print(f"✅ Processing {len(valid_docs)} documents")
    
    # Create and fit BM25 ranker
    print("\n🔄 Training BM25 model...")
    bm25_ranker = BM25Ranker(k1=k1, b=b)
    bm25_ranker.fit(valid_docs)
    
    # Save index
    print(f"\n🔄 Saving BM25 index to {output_path}...")
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(bm25_ranker, f)
        print(f"✅ BM25 index saved successfully")
    except Exception as e:
        print(f"❌ Error saving index: {str(e)}")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ BM25 INDEXING COMPLETE!")
    print("=" * 60)
    print(f"📊 Documents indexed: {len(valid_docs)}")
    print(f"📏 Average document length: {bm25_ranker.avgdl:.2f} tokens")
    print(f"📚 Vocabulary size: {len(bm25_ranker.idf_scores)} unique terms")
    print(f"⚙️  Parameters: k1={k1}, b={b}")
    print(f"\n📁 Output file: {output_path}")
    print(f"💡 Test with: python search_bm25.py --textbook {textbook_id} --query 'test'")


def main():
    parser = argparse.ArgumentParser(
        description="Create BM25 index for textbook chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bm25_indexer.py --textbook intro_to_ml
  python bm25_indexer.py --textbook computer_networks --k1 1.2 --b 0.8
  
BM25 Parameters:
  k1: Controls term frequency saturation (default: 1.5)
      - Higher values give more weight to term frequency
      - Typical range: 1.2 to 2.0
      
  b: Controls document length normalization (default: 0.75)
     - 0 = no length normalization
     - 1 = full length normalization
     - Typical range: 0.5 to 0.9
        """
    )
    
    parser.add_argument(
        '--textbook', '-t',
        required=True,
        help='Textbook ID (must match existing FAISS index)'
    )
    
    parser.add_argument(
        '--indices_dir',
        default='indices',
        help='Directory containing indices (default: indices)'
    )
    
    parser.add_argument(
        '--k1',
        type=float,
        default=1.5,
        help='BM25 k1 parameter (default: 1.5)'
    )
    
    parser.add_argument(
        '--b',
        type=float,
        default=0.75,
        help='BM25 b parameter (default: 0.75)'
    )
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.k1 <= 0:
        print("❌ Error: k1 must be positive")
        sys.exit(1)
    
    if not (0 <= args.b <= 1):
        print("❌ Error: b must be between 0 and 1")
        sys.exit(1)
    
    try:
        create_bm25_index(
            textbook_id=args.textbook,
            indices_dir=args.indices_dir,
            k1=args.k1,
            b=args.b
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())