"""BM25 Variants Comparison Test

This script compares all BM25 implementations on the same corpus and queries,
printing a pretty comparison table to the terminal.

Usage:
    python -m tests.test_bm25_comparison
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bm25.bm25_scorer import BM25
from bm25.bm25_adpt import BM25Adpt
from bm25.bm25l import BM25L
from bm25.bm25plus import BM25Plus
from bm25.bm25t import BM25T

# ANSI color codes
BOLD = "\033[1m"
CYAN = "\033[36m"
RESET = "\033[0m"


# =============================================================================
# CENTRALIZED MODEL CONFIGURATION
# Add new models or modify parameters here
# =============================================================================
MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "BM25": {
        "class": BM25,
        "params": {"k1": 1.5, "b": 0.75},
        "description": "Standard BM25 with inverted index",
    },
    "BM25-Adpt": {
        "class": BM25Adpt,
        "params": {"k1_base": 1.2, "b_base": 0.75, "delta": 0.9},
        "description": "Adaptive parameters based on query characteristics",
    },
    "BM25L": {
        "class": BM25L,
        "params": {"k1": 1.2, "b": 0.75},
        "description": "Logarithmic TF normalization for long docs",
    },
    "BM25+": {
        "class": BM25Plus,
        "params": {"k1": 1.2, "b": 0.75, "delta": 1.0},
        "description": "Lower bound δ for zero-frequency terms",
    },
    "BM25T": {
        "class": BM25T,
        "params": {"k1": 1.2, "b": 0.75, "k3": 1000.0},
        "description": "Two-stage TF transformation",
    },
}


def create_models(corpus: List[str]) -> Dict[str, Any]:
    """Create model instances from configuration.
    
    Args:
        corpus: List of document strings
        
    Returns:
        Dictionary mapping model names to initialized model instances
    """
    return {
        name: config["class"](corpus, **config["params"])
        for name, config in MODELS_CONFIG.items()
    }


def get_model_names() -> List[str]:
    """Get list of model names in defined order."""
    return list(MODELS_CONFIG.keys())


# =============================================================================
# TEST DATA
# =============================================================================
CORPUS = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog jumps over a lazy fox",
    "The fox and the dog are friends",
    "Lazy dogs like to jump over foxes",
    "Brown foxes jump quickly over lazy dogs",
    "Just some unrelated content here with no fox or dog"
]

QUERIES = [
    "quick brown fox",
    "lazy dog",
    "jump",
    "purple elephant"  # Tests zero-frequency handling
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def truncate(text: str, max_len: int = 40) -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def format_score(score: float, width: int = 8) -> str:
    """Format a score with consistent width."""
    return f"{score:>{width}.4f}"


def print_header(title: str, width: int = 80) -> None:
    """Print a styled header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str, width: int = 80) -> None:
    """Print a styled subheader."""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


def get_top_results(bm25_class: Type, corpus: List[str], query: str, k: int = 3) -> List[Tuple[int, float]]:
    """Get top-k results from a BM25 variant."""
    try:
        model = bm25_class(corpus)
        if hasattr(model, 'get_top_k'):
            return model.get_top_k(query, k=k)
        elif hasattr(model, 'score_top_k'):
            return model.score_top_k(query, k=k)
        else:
            return []
    except Exception:
        return [(0, 0.0)]


def get_all_scores(bm25_class: Type, corpus: List[str], query: str) -> List[float]:
    """Get all document scores from a BM25 variant."""
    try:
        model = bm25_class(corpus)
        if hasattr(model, 'get_scores'):
            scores = model.get_scores(query)
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        elif hasattr(model, 'score'):
            scores = model.score(query)
            return list(scores)
        else:
            return [0.0] * len(corpus)
    except Exception:
        return [0.0] * len(corpus)


# =============================================================================
# PRINT FUNCTIONS
# =============================================================================
def print_corpus() -> None:
    """Print the test corpus."""
    print_header("TEST CORPUS")
    for i, doc in enumerate(CORPUS):
        print(f"  [{i}] {truncate(doc, 60)}")
    print(f"\n  Total documents: {len(CORPUS)}")


def print_algorithm_info() -> None:
    """Print information about each BM25 variant."""
    print_header("BM25 ALGORITHM VARIANTS")
    
    for name, config in MODELS_CONFIG.items():
        params_str = ", ".join(f"{k}={v}" for k, v in config["params"].items())
        print(f"\n  {name:12} ({config['class'].__name__})")
        print(f"    Parameters: {params_str}")
        print(f"    {config['description']}")


def print_comparison_table() -> None:
    """Print a comparison table for all queries."""
    print_header("QUERY RESULTS COMPARISON")
    
    models = create_models(CORPUS)
    alg_names = get_model_names()
    
    for query in QUERIES:
        print_subheader(f"Query: '{query}'")
        
        # Get scores from all models
        results: Dict[str, List[Tuple[int, float]]] = {}
        for name, model in models.items():
            if hasattr(model, 'get_scores'):
                scores = model.get_scores(query)
                indexed = [(i, float(scores[i])) for i in range(len(scores))]
                indexed.sort(key=lambda x: x[1], reverse=True)
                results[name] = indexed[:3]
            elif hasattr(model, 'score'):
                scores = model.score(query)
                indexed = [(i, float(scores[i])) for i in range(len(scores))]
                indexed.sort(key=lambda x: x[1], reverse=True)
                results[name] = indexed[:3]
        
        # Print table header
        print(f"\n  {'Rank':>6} | {'Algorithm':>10} | {'Doc':>4} | {'Score':>10} | Document")
        print("  " + "-" * 75)
        
        # Print results
        for rank in range(3):
            for alg_name in alg_names:
                if rank < len(results[alg_name]):
                    doc_idx, score = results[alg_name][rank]
                    doc_preview = truncate(CORPUS[doc_idx], 35)
                    print(f"  {rank+1:>6} | {alg_name:>10} | {doc_idx:>4} | {score:>10.4f} | {doc_preview}")
                else:
                    print(f"  {rank+1:>6} | {alg_name:>10} | {'-':>4} | {'-':>10} | -")
            
            if rank < 2:
                print("  " + "·" * 75)


def print_all_scores_table() -> None:
    """Print a table showing all document scores for each query."""
    print_header("ALL DOCUMENT SCORES")
    
    models = create_models(CORPUS)
    alg_names = get_model_names()
    
    for query in QUERIES:
        print_subheader(f"Query: '{query}'")
        
        # Get all scores
        all_scores: Dict[str, List[float]] = {}
        for name, model in models.items():
            if hasattr(model, 'get_scores'):
                scores = model.get_scores(query)
                all_scores[name] = [float(s) for s in scores]
            elif hasattr(model, 'score'):
                scores = model.score(query)
                all_scores[name] = [float(s) for s in scores]
        
        # Print header
        header = f"  {'Doc':>4} |"
        for name in alg_names:
            header += f" {name:>10} |"
        print(header)
        print("  " + "-" * (6 + 13 * len(alg_names)))
        
        # Print scores for each document
        for doc_idx in range(len(CORPUS)):
            doc_scores = [all_scores[name][doc_idx] for name in alg_names]
            max_score = max(doc_scores) if doc_scores else 0.0
            
            row = f"  {doc_idx:>4} |"
            for name in alg_names:
                score = all_scores[name][doc_idx]
                if score == max_score and max_score > 0:
                    row += f" {CYAN}{BOLD}{score:>10.4f}{RESET} |"
                else:
                    row += f" {score:>10.4f} |"
            print(row)


def print_statistics() -> None:
    """Print summary statistics for each algorithm."""
    print_header("SUMMARY STATISTICS")
    
    models = create_models(CORPUS)
    alg_names = get_model_names()
    
    # Collect stats
    stats: Dict[str, Dict[str, float]] = {
        name: {"min": float('inf'), "max": 0, "sum": 0, "count": 0}
        for name in alg_names
    }
    
    for query in QUERIES:
        for name, model in models.items():
            if hasattr(model, 'get_scores'):
                scores = model.get_scores(query)
                score_list = [float(s) for s in scores]
            elif hasattr(model, 'score'):
                scores = model.score(query)
                score_list = [float(s) for s in scores]
            else:
                continue
                
            stats[name]["min"] = min(stats[name]["min"], min(score_list))
            stats[name]["max"] = max(stats[name]["max"], max(score_list))
            stats[name]["sum"] += sum(score_list)
            stats[name]["count"] += len(score_list)
    
    # Print stats table
    print(f"\n  {'Algorithm':>12} | {'Min Score':>10} | {'Max Score':>10} | {'Avg Score':>10}")
    print("  " + "-" * 52)
    for name in alg_names:
        s = stats[name]
        avg = s["sum"] / s["count"] if s["count"] > 0 else 0
        print(f"  {name:>12} | {s['min']:>10.4f} | {s['max']:>10.4f} | {avg:>10.4f}")


def main() -> None:
    """Run all comparisons and print results."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  BM25 VARIANTS COMPARISON TEST".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    print_corpus()
    print_algorithm_info()
    print_comparison_table()
    print_all_scores_table()
    print_statistics()
    
    print_header("TEST COMPLETE")
    print("\n  All BM25 variants have been compared successfully!")
    print("  Each variant offers different trade-offs for term frequency")
    print("  handling, length normalization, and zero-frequency treatment.\n")


if __name__ == "__main__":
    main()
