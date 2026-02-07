"""Reranker Models Comparison Test

This script compares all available reranking models on the same corpus and queries,
printing a pretty comparison table to the terminal.

Usage:
    python -m tests.test_rerank_comparison
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reranker.reranker import CrossEncoderReranker

# ANSI color codes
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


# =============================================================================
# CENTRALIZED MODEL CONFIGURATION
# Add new models or modify parameters here
# Models should be capable of following the prompt template for scoring.
# =============================================================================
MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "qwen3:0.6b": {
        "display_name": "qwen3-0.6b",
        "max_concurrent": 5,
        "description": "Qwen3 0.6B (small, fast, capable of following scoring prompts)",
    },
    "phi3:mini": {
        "display_name": "phi3-mini",
        "max_concurrent": 5,
        "description": "Phi-3 Mini (fast, lightweight, good instruction following)",
    },
}


def create_rerankers() -> Dict[str, CrossEncoderReranker]:
    """Create reranker instances from configuration.
    
    Returns:
        Dictionary mapping model names to initialized reranker instances
    """
    rerankers = {}
    for model_name, config in MODELS_CONFIG.items():
        rerankers[model_name] = CrossEncoderReranker(
            model_name=model_name,
            max_concurrent=config["max_concurrent"]
        )
    return rerankers


def get_model_names() -> List[str]:
    """Get list of model names in defined order."""
    return list(MODELS_CONFIG.keys())


# =============================================================================
# TEST DATA
# =============================================================================
CORPUS: List[Tuple[str, str]] = [
    ("doc1", "The quick brown fox jumps over the lazy dog"),
    ("doc2", "A quick brown dog jumps over a lazy fox"),
    ("doc3", "The fox and the dog are friends"),
    ("doc4", "Lazy dogs like to jump over foxes"),
    ("doc5", "Brown foxes jump quickly over lazy dogs"),
    ("doc6", "Just some unrelated content here with no fox or dog"),
    ("doc7", "Python is a programming language used for web development and data science"),
    ("doc8", "Authentication is the process of verifying user identity"),
    ("doc9", "Machine learning models require training data and GPUs"),
    ("doc10", "The lazy programmer wrote a quick script to automate the task"),
]

QUERIES = [
    "quick brown fox",
    "lazy dog",
    "jump",
    "authentication",
    "machine learning",
    "purple elephant"  # Tests handling of no relevant documents
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def truncate(text: str, max_len: int = 50) -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def format_score(score: float, width: int = 8) -> str:
    """Format a score with consistent width."""
    return f"{score:>{width}.4f}"


def format_time(time_ms: float, width: int = 10) -> str:
    """Format time in milliseconds with consistent width."""
    return f"{time_ms:>{width}.2f}ms"


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


# =============================================================================
# PRINT FUNCTIONS
# =============================================================================
def print_corpus() -> None:
    """Print the test corpus."""
    print_header("TEST CORPUS")
    for doc_id, doc_text in CORPUS:
        print(f"  [{doc_id}] {truncate(doc_text, 65)}")
    print(f"\n  Total documents: {len(CORPUS)}")


def print_model_info() -> None:
    """Print information about each reranking model."""
    print_header("RERANKING MODELS")
    
    for model_name, config in MODELS_CONFIG.items():
        print(f"\n  {config['display_name']}")
        print(f"    Full name: {model_name}")
        print(f"    Max concurrent: {config['max_concurrent']}")
        print(f"    {config['description']}")


def print_comparison_table() -> None:
    """Print a comparison table for all queries."""
    print_header("QUERY RESULTS COMPARISON")
    
    model_names = get_model_names()
    rerankers = create_rerankers()
    
    for query in QUERIES:
        print_subheader(f"Query: '{query}'")
        
        # Get results from all models
        results: Dict[str, Tuple[List[Tuple[str, float, str]], float]] = {}
        for model_name in model_names:
            reranker = rerankers[model_name]
            start_time = time.perf_counter()
            try:
                top_results = reranker.rerank(query, CORPUS, top_k=5)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                results[model_name] = (top_results, elapsed_ms)
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                results[model_name] = ([], elapsed_ms)
                print(f"  {YELLOW}Warning: Model '{model_name}' failed: {e}{RESET}")
        
        # Print table header
        display_names = [MODELS_CONFIG[m]["display_name"] for m in model_names]
        print(f"\n  {'Rank':>6} | {'Model':>22} | {'Doc':>6} | {'Score':>10} | {'Time':>10} | Document")
        print("  " + "-" * 90)
        
        # Print results for each rank
        for rank in range(5):
            for model_name in model_names:
                display_name = MODELS_CONFIG[model_name]["display_name"]
                top_results, elapsed_ms = results[model_name]
                
                if rank < len(top_results):
                    doc_id, score, doc_text = top_results[rank]
                    doc_preview = truncate(doc_text, 25)
                    time_str = format_time(elapsed_ms, 10) if rank == 0 else " " * 10
                    print(f"  {rank+1:>6} | {display_name:>22} | {doc_id:>6} | {score:>10.4f} | {time_str} | {doc_preview}")
                else:
                    print(f"  {rank+1:>6} | {display_name:>22} | {'N/A':>6} | {'N/A':>10} | {' ' * 10} | -")
            
            if rank < 4:
                print("  " + "·" * 90)


def print_all_scores_table() -> None:
    """Print a table showing scores for top documents across all queries."""
    print_header("TOP DOCUMENTS BY QUERY")
    
    model_names = get_model_names()
    rerankers = create_rerankers()
    
    for query in QUERIES:
        print_subheader(f"Query: '{query}'")
        
        # Get results from all models
        all_results: Dict[str, List[Tuple[str, float, str]]] = {}
        for model_name in model_names:
            reranker = rerankers[model_name]
            try:
                all_results[model_name] = reranker.rerank(query, CORPUS, top_k=3)
            except Exception as e:
                all_results[model_name] = []
                print(f"  {YELLOW}Warning: Model '{model_name}' failed: {e}{RESET}")
        
        # Print header
        header = f"  {'Model':>22} | {'1st':>8} | {'2nd':>8} | {'3rd':>8}"
        print(header)
        print("  " + "-" * 50)
        
        # Print scores for each model
        for model_name in model_names:
            display_name = MODELS_CONFIG[model_name]["display_name"]
            results = all_results.get(model_name, [])
            
            row = f"  {display_name:>22} |"
            for i in range(3):
                if i < len(results):
                    doc_id, score, _ = results[i]
                    row += f" {doc_id}({score:.2f}):{CYAN}{BOLD}{score:>4.2f}{RESET} |"
                else:
                    row += f" {'N/A':>8} |"
            print(row)


def print_statistics() -> None:
    """Print summary statistics for each model."""
    print_header("SUMMARY STATISTICS")
    
    model_names = get_model_names()
    rerankers = create_rerankers()
    
    # Collect stats
    stats: Dict[str, Dict[str, Any]] = {
        name: {
            "scores": [],
            "times": [],
            "errors": 0,
            "successes": 0
        }
        for name in model_names
    }
    
    for query in QUERIES:
        for model_name in model_names:
            reranker = rerankers[model_name]
            start_time = time.perf_counter()
            try:
                results = reranker.rerank(query, CORPUS, top_k=len(CORPUS))
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                stats[model_name]["times"].append(elapsed_ms)
                stats[model_name]["scores"].extend([score for _, score, _ in results])
                stats[model_name]["successes"] += 1
            except Exception:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                stats[model_name]["times"].append(elapsed_ms)
                stats[model_name]["errors"] += 1
    
    # Print stats table
    print(f"\n  {'Model':>22} | {'Avg Time':>10} | {'Min Score':>10} | {'Max Score':>10} | {'Avg Score':>10} | {'Errors':>6}")
    print("  " + "-" * 80)
    
    for model_name in model_names:
        s = stats[model_name]
        display_name = MODELS_CONFIG[model_name]["display_name"]
        
        avg_time = sum(s["times"]) / len(s["times"]) if s["times"] else 0
        min_score = min(s["scores"]) if s["scores"] else 0
        max_score = max(s["scores"]) if s["scores"] else 0
        avg_score = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
        
        time_color = GREEN if avg_time < 500 else (YELLOW if avg_time < 1000 else "")
        print(f"  {display_name:>22} | {time_color}{avg_time:>10.2f}ms{RESET} | {min_score:>10.4f} | {max_score:>10.4f} | {avg_score:>10.4f} | {s['errors']:>6}")


def check_models_available() -> Dict[str, bool]:
    """Check which models are available in Ollama."""
    model_names = get_model_names()
    rerankers = create_rerankers()
    availability = {}
    
    print_header("MODEL AVAILABILITY CHECK")
    
    for model_name in model_names:
        reranker = rerankers[model_name]
        display_name = MODELS_CONFIG[model_name]["display_name"]
        
        try:
            is_available, error_msg = reranker.check_model_available()
            availability[model_name] = is_available
            
            if is_available:
                print(f"  {GREEN}✓{RESET} {display_name} ({model_name})")
            else:
                print(f"  {YELLOW}✗{RESET} {display_name} ({model_name})")
                if error_msg:
                    print(f"    {YELLOW}└─ {error_msg}{RESET}")
        except Exception as e:
            availability[model_name] = False
            print(f"  {YELLOW}✗{RESET} {display_name} ({model_name})")
            print(f"    {YELLOW}└─ Error checking availability: {e}{RESET}")
    
    return availability


def main() -> None:
    """Run all comparisons and print results."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  RERANKER MODELS COMPARISON TEST".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    print_corpus()
    print_model_info()
    
    # Check model availability before running comparisons
    availability = check_models_available()
    
    available_count = sum(1 for v in availability.values() if v)
    if available_count == 0:
        print("\n" + "=" * 80)
        print(f"  {YELLOW}No models are available. Please pull the models first:{RESET}")
        for model_name in get_model_names():
            print(f"    ollama pull {model_name}")
        print("=" * 80 + "\n")
        return
    
    print_comparison_table()
    print_all_scores_table()
    print_statistics()
    
    print_header("TEST COMPLETE")
    print("\n  All reranking models have been compared successfully!")
    print("  Each model offers different trade-offs for:")
    print("  - Speed (Q4 quantized models are faster)")
    print("  - Accuracy (full precision models may be more accurate)")
    print("  - Resource usage (smaller models use less VRAM)\n")


if __name__ == "__main__":
    main()
