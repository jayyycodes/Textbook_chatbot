#!/usr/bin/env python3
"""
Enhanced Search Method Comparison - FAISS vs BM25 vs Hybrid

Compares semantic (FAISS), keyword (BM25), and hybrid approaches.
Includes detailed analytics and recommendations.

Usage:
    python enhanced_compare.py --textbook intro_to_ml --query "neural networks"
    python enhanced_compare.py --textbook intro_to_ml --query "overfitting" --hybrid
    python enhanced_compare.py --textbook intro_to_ml --queries_file test_queries.txt --hybrid
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle
from collections import Counter

# Import search modules
try:
    from search_faiss import MultiTextbookSearcher as FAISSSearcher
    # Try to import enhanced BM25 first, fall back to original
    try:
        from optimized_bm25 import TextbookBM25Searcher as BM25Searcher
        BM25_ENHANCED = True
    except ImportError:
        from search_bm25 import TextbookBM25Searcher as BM25Searcher
        BM25_ENHANCED = False
except ImportError as e:
    print(f"Error: Could not import search modules: {e}")
    sys.exit(1)


class EnhancedSearchComparator:
    """Advanced comparison with hybrid search support."""
    
    def __init__(
        self,
        textbook_id: str,
        indices_dir: str = "indices",
        enable_hybrid: bool = True
    ):
        """Initialize comparator with all search methods."""
        self.textbook_id = textbook_id
        self.indices_dir = indices_dir
        self.enable_hybrid = enable_hybrid
        
        print("🔄 Initializing search systems...")
        
        # Load FAISS
        try:
            self.faiss_searcher = FAISSSearcher(
                textbook_id=textbook_id,
                json_mode=True,
                indices_dir=indices_dir
            )
            print(f"✅ FAISS searcher loaded")
        except Exception as e:
            print(f"❌ Failed to load FAISS: {e}")
            sys.exit(1)
        
        # Load BM25
        try:
            self.bm25_searcher = BM25Searcher(
                textbook_id=textbook_id,
                json_mode=True,
                indices_dir=indices_dir,
                use_enhancements=True if BM25_ENHANCED else False
            )
            enhancement_msg = " (Enhanced)" if BM25_ENHANCED else " (Basic)"
            print(f"✅ BM25 searcher loaded{enhancement_msg}")
        except Exception as e:
            print(f"❌ Failed to load BM25: {e}")
            sys.exit(1)
        
        print("✅ All search systems ready\n")
    
    def normalize_scores(self, results: List[Tuple[float, Dict]]) -> List[Tuple[float, Dict]]:
        """Normalize scores to 0-1 range for fair comparison."""
        if not results:
            return []
        
        scores = [score for score, _ in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [(0.5, doc) for _, doc in results]
        
        normalized = []
        for score, doc in results:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized.append((norm_score, doc))
        
        return normalized
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.4,
        faiss_weight: float = 0.6
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Hybrid search combining BM25 and FAISS.
        
        Args:
            query: Search query
            top_k: Number of results
            bm25_weight: Weight for BM25 scores (0-1)
            faiss_weight: Weight for FAISS scores (0-1)
            
        Returns:
            Combined results with hybrid scores
        """
        # Get results from both methods (more than top_k for better coverage)
        bm25_results = self.bm25_searcher.search(query, top_k * 3)
        faiss_results = self.faiss_searcher.search(query, top_k * 3)
        
        # Normalize scores
        bm25_norm = self.normalize_scores(bm25_results)
        faiss_norm = self.normalize_scores(faiss_results)
        
        # Combine scores
        combined_scores = {}
        
        for score, doc in bm25_norm:
            chunk_id = doc.get('chunk_id')
            combined_scores[chunk_id] = {
                'bm25_score': score * bm25_weight,
                'faiss_score': 0.0,
                'doc': doc
            }
        
        for score, doc in faiss_norm:
            chunk_id = doc.get('chunk_id')
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['faiss_score'] = score * faiss_weight
            else:
                combined_scores[chunk_id] = {
                    'bm25_score': 0.0,
                    'faiss_score': score * faiss_weight,
                    'doc': doc
                }
        
        # Calculate final scores
        final_results = []
        for chunk_id, data in combined_scores.items():
            hybrid_score = data['bm25_score'] + data['faiss_score']
            final_results.append((hybrid_score, data['doc']))
        
        # Sort by hybrid score
        final_results.sort(key=lambda x: x[0], reverse=True)
        
        return final_results[:top_k]
    
    def compare_all_methods(
        self,
        query: str,
        top_k: int = 5,
        show_details: bool = True
    ) -> Dict[str, Any]:
        """Compare all three methods: BM25, FAISS, and Hybrid."""
        
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        
        results = {}
        
        # 1. BM25 Search
        print("\n🔍 Searching with BM25 (Keyword)...")
        bm25_start = time.time()
        try:
            bm25_results = self.bm25_searcher.search(query, top_k)
            bm25_time = time.time() - bm25_start
            results['bm25'] = {
                'results': bm25_results,
                'time_ms': round(bm25_time * 1000, 2),
                'error': None
            }
            print(f"   ✅ Found {len(bm25_results)} results in {bm25_time*1000:.2f}ms")
        except Exception as e:
            results['bm25'] = {'results': [], 'time_ms': 0, 'error': str(e)}
            print(f"   ❌ Error: {e}")
        
        # 2. FAISS Search
        print("🔍 Searching with FAISS (Semantic)...")
        faiss_start = time.time()
        try:
            faiss_results = self.faiss_searcher.search(query, top_k)
            faiss_time = time.time() - faiss_start
            results['faiss'] = {
                'results': faiss_results,
                'time_ms': round(faiss_time * 1000, 2),
                'error': None
            }
            print(f"   ✅ Found {len(faiss_results)} results in {faiss_time*1000:.2f}ms")
        except Exception as e:
            results['faiss'] = {'results': [], 'time_ms': 0, 'error': str(e)}
            print(f"   ❌ Error: {e}")
        
        # 3. Hybrid Search
        if self.enable_hybrid:
            print("🔍 Searching with Hybrid (Combined)...")
            hybrid_start = time.time()
            try:
                hybrid_results = self.hybrid_search(query, top_k)
                hybrid_time = time.time() - hybrid_start
                results['hybrid'] = {
                    'results': hybrid_results,
                    'time_ms': round(hybrid_time * 1000, 2),
                    'error': None
                }
                print(f"   ✅ Found {len(hybrid_results)} results in {hybrid_time*1000:.2f}ms")
            except Exception as e:
                results['hybrid'] = {'results': [], 'time_ms': 0, 'error': str(e)}
                print(f"   ❌ Error: {e}")
        
        # Analyze results
        comparison = {
            'query': query,
            'top_k': top_k,
            'methods': self._format_all_results(results),
            'analysis': self._analyze_all_results(results),
            'recommendation': self._generate_recommendation(query, results)
        }
        
        if show_details:
            self._print_detailed_comparison(comparison)
        
        return comparison
    
    def _format_all_results(self, results: Dict) -> Dict:
        """Format results from all methods."""
        formatted = {}
        
        for method, data in results.items():
            formatted[method] = {
                'count': len(data['results']),
                'time_ms': data['time_ms'],
                'error': data['error'],
                'results': [
                    {
                        'rank': rank,
                        'score': round(score, 4),
                        'chunk_id': doc.get('chunk_id', 'Unknown'),
                        'text_preview': doc.get('text', '')[:150] + "..."
                    }
                    for rank, (score, doc) in enumerate(data['results'], 1)
                ]
            }
        
        return formatted
    
    def _analyze_all_results(self, results: Dict) -> Dict:
        """Comprehensive analysis of all methods."""
        
        # Extract chunk IDs from each method
        chunk_sets = {}
        for method, data in results.items():
            if data['results']:
                chunk_sets[method] = set(doc.get('chunk_id') for _, doc in data['results'])
            else:
                chunk_sets[method] = set()
        
        analysis = {
            'overlap': {},
            'performance': {},
            'diversity': {}
        }
        
        # Pairwise overlap
        methods = list(chunk_sets.keys())
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                set1 = chunk_sets[method1]
                set2 = chunk_sets[method2]
                common = set1 & set2
                
                if set1 or set2:
                    overlap_pct = len(common) / max(len(set1), len(set2)) * 100
                else:
                    overlap_pct = 0
                
                analysis['overlap'][f"{method1}_vs_{method2}"] = {
                    'common': len(common),
                    'overlap_pct': round(overlap_pct, 2)
                }
        
        # Performance comparison
        for method, data in results.items():
            analysis['performance'][method] = {
                'time_ms': data['time_ms'],
                'result_count': len(data['results']),
                'has_error': data['error'] is not None
            }
        
        # Diversity (unique chunks per method)
        all_chunks = set()
        for chunks in chunk_sets.values():
            all_chunks.update(chunks)
        
        for method, chunks in chunk_sets.items():
            unique = chunks - set().union(*[chunk_sets[m] for m in chunk_sets if m != method])
            analysis['diversity'][method] = {
                'unique_chunks': len(unique),
                'total_unique_pct': round(len(unique) / len(all_chunks) * 100, 2) if all_chunks else 0
            }
        
        return analysis
    
    def _generate_recommendation(self, query: str, results: Dict) -> Dict:
        """Generate intelligent recommendation based on query and results."""
        
        recommendation = {
            'best_method': None,
            'confidence': 'low',
            'reasoning': [],
            'query_type': self._classify_query(query)
        }
        
        # Classify query type
        query_type = recommendation['query_type']
        
        # Rule-based recommendation
        if query_type == 'keyword':
            recommendation['best_method'] = 'bm25'
            recommendation['confidence'] = 'high'
            recommendation['reasoning'].append(
                "Query contains specific technical terms - BM25 excels at exact matching"
            )
        
        elif query_type == 'conceptual':
            recommendation['best_method'] = 'faiss'
            recommendation['confidence'] = 'high'
            recommendation['reasoning'].append(
                "Query is conceptual/semantic - FAISS better captures meaning"
            )
        
        elif query_type == 'mixed':
            recommendation['best_method'] = 'hybrid'
            recommendation['confidence'] = 'high'
            recommendation['reasoning'].append(
                "Query has both keywords and concepts - Hybrid approach recommended"
            )
        
        else:  # question
            recommendation['best_method'] = 'hybrid'
            recommendation['confidence'] = 'medium'
            recommendation['reasoning'].append(
                "Natural question - Hybrid approach balances keyword and semantic matching"
            )
        
        # Check if methods actually performed well
        if results.get(recommendation['best_method'], {}).get('error'):
            recommendation['best_method'] = 'faiss'  # Fallback
            recommendation['confidence'] = 'low'
            recommendation['reasoning'].append("Recommended method failed - using FAISS as fallback")
        
        return recommendation
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for better recommendation."""
        query_lower = query.lower()
        
        # Technical/keyword indicators
        technical_terms = ['algorithm', 'function', 'formula', 'equation', 'method']
        has_technical = any(term in query_lower for term in technical_terms)
        
        # Question indicators
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'explain']
        is_question = any(query_lower.startswith(qw) for qw in question_words)
        is_question = is_question or '?' in query
        
        # Conceptual indicators
        conceptual_words = ['understand', 'concept', 'idea', 'difference', 'compare']
        is_conceptual = any(word in query_lower for word in conceptual_words)
        
        # Classification logic
        if has_technical and not is_question:
            return 'keyword'
        elif is_conceptual or (is_question and len(query.split()) > 6):
            return 'conceptual'
        elif is_question:
            return 'question'
        else:
            return 'mixed'
    
    def _print_detailed_comparison(self, comparison: Dict):
        """Print comprehensive comparison results."""
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE COMPARISON RESULTS")
        print("=" * 70)
        
        # Query info
        print(f"\n📝 Query: {comparison['query']}")
        print(f"🏷️  Query Type: {comparison['recommendation']['query_type'].upper()}")
        
        # Performance metrics
        print("\n⏱️  PERFORMANCE:")
        methods = comparison['methods']
        for method, data in methods.items():
            emoji = "🔤" if method == 'bm25' else "🧠" if method == 'faiss' else "🔀"
            status = "✅" if not data['error'] else "❌"
            print(f"  {emoji} {method.upper()}: {data['time_ms']}ms | {data['count']} results {status}")
        
        # Overlap analysis
        print("\n🔄 RESULTS OVERLAP:")
        overlap = comparison['analysis']['overlap']
        for pair, data in overlap.items():
            method1, method2 = pair.split('_vs_')
            print(f"  {method1.upper()} ∩ {method2.upper()}: {data['common']} chunks ({data['overlap_pct']:.1f}%)")
        
        # Diversity
        print("\n🎨 RESULT DIVERSITY (Unique Contributions):")
        diversity = comparison['analysis']['diversity']
        for method, data in diversity.items():
            print(f"  {method.upper()}: {data['unique_chunks']} unique chunks")
        
        # Recommendation
        print("\n🎯 RECOMMENDATION:")
        rec = comparison['recommendation']
        print(f"  Best Method: {rec['best_method'].upper()}")
        print(f"  Confidence: {rec['confidence'].upper()}")
        print(f"  Reasoning:")
        for reason in rec['reasoning']:
            print(f"    • {reason}")
        
        # Top 3 from each method
        print("\n🥇 TOP 3 RESULTS FROM EACH METHOD:")
        
        for method in ['bm25', 'faiss', 'hybrid']:
            if method not in methods or methods[method]['error']:
                continue
            
            emoji = "🔤" if method == 'bm25' else "🧠" if method == 'faiss' else "🔀"
            print(f"\n  {emoji} {method.upper()}:")
            
            for result in methods[method]['results'][:3]:
                print(f"    {result['rank']}. [{result['chunk_id']}] Score: {result['score']:.4f}")
                print(f"       {result['text_preview']}")
        
        print("\n" + "=" * 70)
    
    def evaluate_with_ground_truth(
        self,
        queries_with_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate all methods against ground truth.
        
        Args:
            queries_with_truth: List of dicts with 'query' and 'relevant_chunks'
        """
        print("\n🧪 EVALUATION WITH GROUND TRUTH")
        print("=" * 70)
        
        all_metrics = {
            'bm25': {'precision@5': [], 'recall@10': [], 'mrr': []},
            'faiss': {'precision@5': [], 'recall@10': [], 'mrr': []},
            'hybrid': {'precision@5': [], 'recall@10': [], 'mrr': []}
        }
        
        for i, item in enumerate(queries_with_truth, 1):
            query = item['query']
            relevant = set(item['relevant_chunks'])
            
            print(f"\n[{i}/{len(queries_with_truth)}] {query}")
            
            # Test each method
            for method_name in ['bm25', 'faiss', 'hybrid']:
                try:
                    if method_name == 'bm25':
                        results = self.bm25_searcher.search(query, 10)
                    elif method_name == 'faiss':
                        results = self.faiss_searcher.search(query, 10)
                    else:
                        results = self.hybrid_search(query, 10)
                    
                    retrieved = [doc.get('chunk_id') for _, doc in results]
                    
                    # Precision@5
                    top5 = set(retrieved[:5])
                    p5 = len(top5 & relevant) / 5 if relevant else 0
                    all_metrics[method_name]['precision@5'].append(p5)
                    
                    # Recall@10
                    top10 = set(retrieved[:10])
                    r10 = len(top10 & relevant) / len(relevant) if relevant else 0
                    all_metrics[method_name]['recall@10'].append(r10)
                    
                    # MRR
                    mrr = 0.0
                    for rank, chunk_id in enumerate(retrieved, 1):
                        if chunk_id in relevant:
                            mrr = 1.0 / rank
                            break
                    all_metrics[method_name]['mrr'].append(mrr)
                    
                    print(f"  {method_name.upper()}: P@5={p5:.3f}, R@10={r10:.3f}, MRR={mrr:.3f}")
                
                except Exception as e:
                    print(f"  {method_name.upper()}: Error - {e}")
        
        # Calculate averages
        summary = {}
        for method, metrics in all_metrics.items():
            summary[method] = {
                metric: round(sum(values) / len(values), 4) if values else 0.0
                for metric, values in metrics.items()
            }
        
        print("\n📊 AVERAGE METRICS:")
        print("-" * 70)
        print(f"{'Method':<10} {'P@5':<10} {'R@10':<10} {'MRR':<10}")
        print("-" * 70)
        for method, metrics in summary.items():
            print(f"{method.upper():<10} {metrics['precision@5']:<10.4f} {metrics['recall@10']:<10.4f} {metrics['mrr']:<10.4f}")
        
        return summary


def load_queries_from_file(file_path: str) -> List[str]:
    """Load queries from text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_ground_truth(file_path: str) -> List[Dict]:
    """Load ground truth from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)['queries']


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced search comparison with hybrid support"
    )
    
    parser.add_argument('--textbook', '-t', required=True,
                       help='Textbook ID')
    parser.add_argument('--query', '-q',
                       help='Single query')
    parser.add_argument('--queries_file', '-f',
                       help='File with queries')
    parser.add_argument('--ground_truth', '-g',
                       help='JSON file with ground truth')
    parser.add_argument('--top_k', '-k', type=int, default=5,
                       help='Results per query')
    parser.add_argument('--hybrid', action='store_true',
                       help='Enable hybrid search')
    parser.add_argument('--indices_dir', default='indices',
                       help='Indices directory')
    parser.add_argument('--output', '-o',
                       help='Save results to JSON')
    
    args = parser.parse_args()
    
    try:
        comparator = EnhancedSearchComparator(
            textbook_id=args.textbook,
            indices_dir=args.indices_dir,
            enable_hybrid=args.hybrid
        )
        
        results = None
        
        if args.ground_truth:
            # Evaluation mode
            queries_with_truth = load_ground_truth(args.ground_truth)
            results = comparator.evaluate_with_ground_truth(queries_with_truth)
        
        elif args.query:
            # Single query mode
            results = comparator.compare_all_methods(args.query, args.top_k)
        
        elif args.queries_file:
            # Multiple queries mode
            queries = load_queries_from_file(args.queries_file)
            results = []
            for query in queries:
                result = comparator.compare_all_methods(query, args.top_k, show_details=False)
                results.append(result)
                print(f"✅ {query[:50]}... - Recommended: {result['recommendation']['best_method']}")
        
        else:
            print("❌ Provide --query, --queries_file, or --ground_truth")
            return 1
        
        # Save results
        if args.output and results:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {args.output}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())