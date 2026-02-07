# Workspace RAG Search Tool

A powerful semantic code search tool for local repositories using **RAG (Retrieval-Augmented Generation)** with ChromaDB and local Ollama embeddings. Search your codebase using natural language queries with intelligent hybrid scoring (semantic + keyword matching).

## Features

- 🔍 **Semantic Search** - Find code by meaning, not just keywords
- 🚀 **RRF Hybrid Fusion** - Uses Reciprocal Rank Fusion to combine semantic + BM25 rankings for superior recall
- 🎯 **Cross-Encoder Reranking** - Optional LLM-based reranking for +25-40% relevance improvement
- 🎭 **MMR Diversity Reranking** - Maximal Marginal Relevance reduces duplicate results from the same files
- 🧠 **HyDE Query Expansion** - Hypothetical Document Embeddings for better retrieval on vague queries
- 🧮 **Code-Optimized BM25** - Custom tokenization handling snake_case, camelCase, and kebab-case identifiers
- 📊 **Performance Metrics** - Detailed latency, diversity, and coverage statistics for observability
- 📁 **Smart Indexing** - Incremental updates, only processes changed files
- 🦙 **Local Embeddings** - Uses Ollama for 100% local, private embeddings
- 🚫 **Git-Aware** - Automatically respects `.gitignore` rules
- 🧠 **Content Detection** - Automatically skips binary files
- ⚡ **Fast** - Persistent vector store with ChromaDB
- 💾 **Query Caching** - LRU cache for sub-second repeated searches
- 🎯 **Configurable** - Customizable chunking, extensions, and filters

## When to Use This Tool

This tool is designed for scenarios where **privacy, control, and local execution** are priorities:

| Scenario | Why This Tool Fits |
|----------|-------------------|
| **🔒 Air-Gapped / Offline Development** | Works entirely without internet connectivity. All models (embeddings, reranking) run locally via Ollama—no data ever leaves your machine. |
| **🏢 Enterprise Codebases** | Keep sensitive source code and search indexes completely within your infrastructure. No third-party APIs, no external data processing, full compliance with security policies. |
| **⚡ CI/CD Pipelines** | Fast, local semantic search for automated code review, documentation generation, or test discovery. Runs on self-hosted runners without external API dependencies or rate limits. |
| **🔐 Privacy-Conscious Projects** | Ideal for proprietary code, personal projects, or any situation where you don't want your codebase sent to cloud-based embedding services. |

## Prerequisites

1. **Python 3.9+**
2. **Ollama**

### Install Models

```bash
# Required: Embedding model
ollama pull nomic-embed-text

# Optional: Reranking model (for +25-40% relevance improvement)
ollama pull phi3:mini

# Alternative reranking model (faster, lighter):
# ollama pull qwen3:0.6b
```

**Recommended Models for Reranking:**
| Model | Size | VRAM | Best For |
|-------|------|------|----------|
| `qwen3:0.6b` | ~522MB | ~1GB | Default, good quality/performance balance |
| `phi3:mini` | ~2.2 GB | ~1GB | Faster inference, lightweight |

- `nomic-embed-text` is default embedding model (configure your own in `ollama_config.py`)
- Reranking models are optional but significantly improve result quality

## Installation

### From PyPI

```bash
pip install workspace-rag-tool
```

### From Source

```
git clone https://github.com/kipwoker/workspace-rag-search.git
cd workspace-rag-search

python -m venv .venv
# .venv\Scripts\Activate     # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

## Quick Start

```python
from workspace_rag_search_tool import WorkspaceRagSearchTool

# Initialize the tool with your repository path
tool = WorkspaceRagSearchTool("/path/to/your/codebase")

# Search for code using natural language
results = tool.search_workspace("authentication middleware", limit=5)
print(results)

# Filter results by file path
results = tool.search_workspace("database connection", path_filter="models")

# Control content preview length (default: no truncation)
results = tool.search_workspace("authentication", preview_window=1000)

# Get index statistics
stats = tool.get_index_stats()
print(stats)

# Refresh the index after code changes
tool.refresh_index()

# Get cache statistics (hit rate, size, etc.)
cache_stats = tool.get_cache_stats()
print(cache_stats)

# Clear the query cache if needed
tool.clear_cache()
```

## CLI Example

The repository includes `example.py` - a ready-to-use CLI tool that reads search parameters from a `example_in.json` file.

### Running the Example

```bash
> python _examples/example.py _examples/example_in.json
```

<details>
<summary>example output</summary>

```
> python _examples/example.py _examples/example_in.json
============================================================
Workspace RAG Search Tool
============================================================

📁 Workspace: .
🔍 Query: compute file hash defintion
🔧 Path filter: utils
🔢 Limit: 3
📏 Preview window: 1500

⚙️  Initializing indexer...
   (This may take a while for the first run)

2026-02-07 19:01:28,639 [INFO] ◦ workspace_rag_search_tool ◦ Initializing workspace search index for: .../workspace-rag-search
2026-02-07 19:01:28,958 [INFO] ◦ workspace_rag_search_tool ◦ Created new collection: workspace_code_index
2026-02-07 19:01:32,428 [INFO] ◦ workspace_rag_search_tool ◦ Found 38 files to index (39 from workspace, 1 binary/non-text skipped)
2026-02-07 19:01:32,448 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 10% (4/38 files, 629.0B/291.3KB)
2026-02-07 19:01:32,450 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 21% (8/38 files, 18.0KB/291.3KB)
2026-02-07 19:01:32,451 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 31% (12/38 files, 65.3KB/291.3KB)
2026-02-07 19:01:34,761 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 42% (16/38 files, 106.9KB/291.3KB)
2026-02-07 19:01:34,762 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 50% (19/38 files, 121.9KB/291.3KB)
2026-02-07 19:01:36,060 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 60% (23/38 files, 155.5KB/291.3KB)
2026-02-07 19:01:36,062 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 71% (27/38 files, 172.9KB/291.3KB)
2026-02-07 19:01:36,063 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 81% (31/38 files, 216.5KB/291.3KB)
2026-02-07 19:01:37,408 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 92% (35/38 files, 238.4KB/291.3KB)
2026-02-07 19:01:37,410 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 100% (38/38 files, 291.3KB/291.3KB)
2026-02-07 19:01:38,543 [INFO] ◦ reranker.reranker ◦ Initialized CrossEncoderReranker with model=phi3:mini, max_concurrent=5
2026-02-07 19:01:40,845 [INFO] ◦ workspace_rag_search_tool ◦ Reranker initialized with model: phi3:mini
2026-02-07 19:01:40,845 [INFO] ◦ workspace_rag_search_tool ◦ Query cache initialized (max_size=100, ttl=none)
2026-02-07 19:01:40,845 [INFO] ◦ workspace_rag_search_tool ◦ Workspace index ready!
✅ Index ready!

🛠️  tool → → → ◦ [search_workspace] ◦ {"query": "compute file hash defintion", "limit": 3, "path_filter": "utils", "preview_window": 1500}
2026-02-07 19:01:45,328 [INFO] ◦ mmr.mmr ◦ Initialized MMRReranker (lambda=0.60, max_file_chunks=2, file_penalty=0.10)
2026-02-07 19:01:45,328 [INFO] ◦ mmr.mmr ◦ MMR reranking complete: selected 3 diverse results from 3 candidates
📄 tool ← ← ← ◦ [search_workspace] ◦
{
  "status": "success",
  "count": 3,
  "rrf_k": 60,
  "coverage": {
    "semantic_only": 0,
    "bm25_only": 0,
    "both_methods": 3
  },
  "results": "Found 3 relevant snippets using RRF + Reranking + MMR:

--- Result 1 Final: 0.54 | Rerank: #1 | RRF: 0.0325 | Semantic: #1 | BM25: #2 (semantic: 0.646, bm25: 5.708) ---
[File: utils/file_utils.py]
rue if file can be read as text, False otherwise
    \"\"\"
    try:
        with open(file_path, \"rb\") as f:
            raw = f.read(sample_size)

        if not raw:
            return True

        raw.decode(\"utf-8\", errors=\"strict\")
        return True
    except (UnicodeDecodeError, IOError, OSError, PermissionError):
        return False


def compute_file_hash(file_path: Path) -> str:
    \"\"\"Compute a hash of the file content for change detection.

    Args:
        file_path: Path to the file

    Returns:
        MD5 hash of the file content
    \"\"\"
    try:
        with open(file_path, \"rb\") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.warning(\"Could not hash file %s: %s\", file_path, e)
        return \"\"


def format_size(size_bytes: int) -> str:
    \"\"\"Format bytes to human readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human readable string (e.

--- Result 2 Final: 0.1167 | Rerank: #2 | RRF: 0.0323 | Semantic: #3 | BM25: #1 (semantic: 0.526, bm25: 6.108) ---
[File: utils/__init__.py]
\"\"\"Utility modules for workspace_rag_search_tool.

This package contains helper functions and utilities that are not
directly related to RAG functionality but support file operations,
gitignore parsing, and path handling.
\"\"\"

from .file_utils import is_text_file, compute_file_hash, format_size
from .gitignore_utils import GitignoreParser
from .path_utils import PathResolver

__all__ = [
    \"is_text_file\",
    \"compute_file_hash\",
    \"format_size\",
    \"GitignoreParser\",
    \"PathResolver\",
]

--- Result 3 Final: -0.4116 | Rerank: #3 | RRF: 0.032 | Semantic: #2 | BM25: #3 (semantic: 0.578, bm25: 3.376) ---
[File: utils/file_utils.py]
\"\"\"File utility functions for workspace indexing.

This module provides helper functions for file operations including
text detection, hashing, and size formatting.
\"\"\"

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_text_file(file_path: Path, sample_size: int = 8192) -> bool:
    \"\"\"Check if a file is text-readable by attempting to decode it as UTF-8.

    This is more reliable than extension-based filtering as it handles:
    - Files without extensions
    - Files with wrong extensions
    - Binary files that happen to have text extensions
    - Various text encodings

    Args:
        file_path: Path to the file to check
        sample_size: Number of bytes to read for detection (default 8KB)

    Returns:
        True if file can be read as text, False otherwise
    \"\"\"
    try:
        with open(file_path, \"rb\") as f:
            raw = f.read(sample_size)

        if not raw:
            return True",
  "query": "compute file hash defintion",
  "reranking": {
    "enabled": true,
    "model": "phi3:mini",
    "latency_ms": 3918.75,
    "candidates": 3
  },
  "mmr": {
    "enabled": true,
    "lambda": 0.6,
    "max_file_chunks": 2,
    "latency_ms": 2.15,
    "candidates": 20
  },
  "metrics": {
    "latency": {
      "semantic_search_ms": 538.44,
      "bm25_build_ms": 22.95,
      "bm25_score_ms": 0.0,
      "rrf_fusion_ms": 0.0,
      "rerank_ms": 3918.75,
      "fetch_results_ms": 1.0,
      "total_ms": 4483.29,
      "mmr_ms": 2.15
    },
    "diversity": {
      "unique_files": 2,
      "file_diversity_ratio": 0.667,
      "score_range": 0.9516,
      "score_std": 0.3893,
      "method_agreement": 1.0
    },
    "coverage": {
      "semantic_hits": 3,
      "bm25_hits": 3,
      "both_hits": 3,
      "total_results": 3
    }
  },
  "latency_ms": 4483.29,
  "cached": false
}
```
</details>

## Configuration

Edit `ollama_config.py` to customize behavior. The main configuration is done through the `RAGConfig` dataclass:

```python
from ollama_config import RAGConfig, RAG_CONFIG_DEFAULT

# Start with defaults and customize
config = RAG_CONFIG_DEFAULT
config.rerank_model = "phi3:mini"  # Use faster model
config.rerank_top_k = 10           # Rerank fewer documents

# Or create a custom config from scratch
from ollama_config import RAGConfig, BM25Implementation

custom_config = RAGConfig(
    vector_store_path="./.vectorstore",
    embedding_model="nomic-embed-text",
    chunk_size=1000,              # Characters per chunk
    chunk_overlap=200,            # Overlap between chunks
    bytes_limit=100000,           # Max file size to index
    max_concurrent_requests=10,   # Concurrent embedding requests
    embedding_batch_size=32,      # Texts per embedding API call
    bm25_implementation="plus",   # BM25 variant
    rerank_enabled=True,
    rerank_model="phi3:mini",
    rerank_top_k=20,
    rerank_max_concurrent=5,
    cache_enabled=True,           # Enable query result caching
    cache_max_size=100,           # Maximum cache entries
    cache_ttl_seconds=300,        # TTL in seconds (None = no expiration)
)
```

### BM25 Implementations

Choose from several BM25 variants:
- `"standard"` - Classic BM25
- `"plus"` - BM25+ with lower bound for zero-frequency terms (default)
- `"l"` - BM25L with logarithmic TF normalization
- `"t"` - BM25T with two-stage TF transformation
- `"adpt"` - BM25-Adpt with adaptive parameter tuning

## Advanced Usage

### Filter by File Extensions

```python
# Only index Python and JavaScript files
tool = WorkspaceRagSearchTool(
    "/path/to/code",
    include_extensions={".py", ".js", ".ts"}
)

# Exclude minified files
tool = WorkspaceRagSearchTool(
    "/path/to/code",
    exclude_extensions={".min.js", ".map"}
)
```

### Force Reindex

```python
# Delete and recreate the vector store
tool = WorkspaceRagSearchTool("/path/to/code", force_reindex=True)
```

### Content Preview Window

Control how much of each result's content is displayed in search results:

```python
# Increase preview window for more context
results = tool.search_workspace("authentication middleware", preview_window=1000)

# Show full content without truncation (preview_window=None) (default)
results = tool.search_workspace("authentication middleware", preview_window=None)
```

### Search Output Format

The `search_workspace()` method returns a JSON string with the following structure:

```json
{
  "status": "success",
  "count": 3,
  "rrf_k": 60,
  "coverage": {
    "semantic_only": 0,
    "bm25_only": 0,
    "both_methods": 3
  },
  "results": "Found 3 relevant snippets using RRF + Reranking:\n\n--- Result 1 Final: 0.85 | Rerank: #1 | RRF: 0.0325 | Semantic: #1 | BM25: #2 (semantic: 0.646, bm25: 5.708) ---\n[File: utils/file_utils.py]\n...",
  "query": "compute file hash defintion",
  "reranking": {
    "enabled": true,
    "model": "phi3:mini",
    "latency_ms": 2136.0,
    "candidates": 3
  }
}
```

> [!WARNING]
> The `preview_window` parameter limits how many characters are displayed from the start of each result. If your search term appears later in the chunk, it may not be visible in the truncated preview. Set `preview_window=None` (default) to display the full chunk content and ensure matches are always visible.

### Query Caching

The tool includes an **LRU (Least Recently Used) cache** for search query results, providing sub-second responses for repeated searches.

**How it works:**
- Cache keys are based on query parameters (query string, limit, path_filter, rrf_k, rerank_enabled)
- Results are cached after the first search
- Subsequent identical queries return instantly from cache
- Cache is automatically cleared when the index is refreshed

**Cache Options:**
- `cache_enabled`: Toggle caching on/off (default: `True`)
- `cache_max_size`: Maximum number of cached queries (default: `100`)
- `cache_ttl_seconds`: Time-to-live for entries. `None` means no expiration (default: `None`)

**Managing the Cache:**
```python
# Get cache statistics
stats = tool.get_cache_stats()
print(stats)
# Output: {"status": "success", "cache": {"hits": 42, "misses": 10, "hit_rate": 0.8077, ...}}

# Clear all cached queries
tool.clear_cache()

# Clear specific queries matching a pattern
tool.clear_cache("authentication")
```

## How It Works

1. **Indexing Phase:**
   - Scans workspace files (respecting `.gitignore`)
   - Filters binary files using content detection
   - Chunks files with configurable overlap
   - Generates embeddings using local Ollama model
   - Stores in ChromaDB vector database

2. **Search Phase:**
   - Converts query to embedding vector
   - **Independent Retrieval**: Retrieves top-k results from both semantic search (ChromaDB) and BM25 (full corpus)
   - **RRF Fusion**: Combines rankings using Reciprocal Rank Fusion formula: `score(d) = Σ 1/(k + rank_d)`
   - Documents appearing in both result lists get boosted scores
   - BM25 uses code-optimized tokenization for better identifier matching
   - Returns ranked results with coverage statistics (semantic-only, BM25-only, both methods)

### Cross-Encoder Reranking

Cross-encoder reranking uses an LLM to score query-document relevance, providing +25-40% improvement in result quality. It processes the top-k RRF results and reorders them based on fine-grained semantic understanding.

**How it works:**
1. RRF fusion produces initial ranked list from semantic + BM25
2. Cross-encoder scores each query-document pair
3. Results are reordered by the new relevance scores
4. Final results show both RRF and reranked scores

**Configuration**
- Check the `ollama_config.py`

**Performance Presets:**
- `RAG_CONFIG_DEFAULT`: Reranking, caching, and metrics enabled
- `RAG_CONFIG_FAST`: Reranking disabled, caching with 60s TTL, metrics disabled
- `RAG_CONFIG_CONSERVATIVE`: Lower concurrency, larger cache with 600s TTL, metrics enabled

### Performance Metrics

When `metrics_enabled=True` (default in `RAG_CONFIG_DEFAULT`), detailed performance statistics are included in search results:

**Latency Metrics:**
- `semantic_search_ms` - ChromaDB vector search time
- `bm25_build_ms` - BM25 index construction time
- `bm25_score_ms` - BM25 scoring time
- `rrf_fusion_ms` - RRF fusion computation time
- `rerank_ms` - Cross-encoder reranking time (if enabled)
- `fetch_results_ms` - Document retrieval time
- `total_ms` - Overall search latency

**Diversity Metrics:**
- `unique_files` - Number of unique files in results
- `file_diversity_ratio` - Ratio of unique files to total results (0-1)
- `score_range` - Difference between highest and lowest scores
- `score_std` - Standard deviation of scores (indicates result spread)
- `method_agreement` - Fraction of results found by both semantic and BM25

**Coverage Metrics:**
- `semantic_hits` / `bm25_hits` - Results from each retrieval method
- `both_hits` - Results found by both methods
- `total_results` - Total number of results returned



**Choosing a Reranking Model:**
- `qwen3:0.6b`: Small, capable model with good quality/performance balance
- `phi3:mini` (default): Lightweight, faster inference, good for quick reranking on limited hardware

### MMR Diversity Reranking

**Maximal Marginal Relevance (MMR)** reduces result duplication by explicitly trading off relevance against diversity. This is especially useful for code search where multiple chunks from the same file can dominate results.

**MMR Formula:**
```
MMR_score = λ * relevance - (1-λ) * max_similarity_to_selected
```

Where:
- `λ` (lambda): Trade-off parameter (0-1)
  - `1.0` = Pure relevance (no diversity)
  - `0.5` = Balanced (default)
  - `0.0` = Pure diversity (ignore relevance)

**How it works:**
1. Takes candidates from RRF (or reranking if enabled)
2. Greedily selects documents that maximize the MMR score
3. Uses embeddings to compute semantic similarity
4. Optionally limits chunks per file for better file-level diversity

**Performance Presets:**
- `RAG_CONFIG_DEFAULT`: MMR enabled with lambda=0.6, max 2 chunks per file
- `RAG_CONFIG_FAST`: MMR disabled for speed
- `RAG_CONFIG_CONSERVATIVE`: MMR enabled with lambda=0.7, max 1 chunk per file

**When to use MMR:**
- ✅ Results show multiple chunks from the same file
- ✅ You want broader code coverage across files
- ✅ Exploring a codebase (not looking for specific implementations)
- ❌ Looking for the most relevant single implementation
- ❌ Query is very specific (e.g., "function foo in bar.py")

### HyDE Query Expansion

**Hypothetical Document Embeddings (HyDE)** improves retrieval on complex or vague queries by generating a hypothetical answer document and using it for semantic search.

**How it works:**
1. User query (e.g., "compute file hash defintion") is sent to an LLM
2. LLM generates a hypothetical code snippet or documentation excerpt
3. This hypothetical document is used for semantic search instead of the original query
4. The richer context often matches actual code more closely than vague user queries

**HyDE Reranking Strategies:**
When HyDE is enabled, you can choose how the reranker uses the generated document:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `"hyde"` (default) | Use HyDE document for reranking. Aligns reranker with semantic search. | Most cases - ensures consistency |
| `"original"` | Use original query for reranking. Legacy behavior. | When you want reranker to judge based on exact query terms |
| `"combined"` | Use both: `"Query: X\n\nHypothetical Answer:\nY"` | When you want the benefits of both approaches |
| `"skip"` | Skip reranking when HyDE is enabled | Faster results, trust semantic search completely |

**Why this matters:** Without proper alignment, semantic search may find the right document using the HyDE context, but the reranker (using the original vague query) may incorrectly deprioritize it. The `"hyde"` strategy ensures both stages use the same rich context.

**Performance Presets:**
- `RAG_CONFIG_DEFAULT`: HyDE enabled with `hyde` reranking strategy
- `RAG_CONFIG_FAST`: HyDE disabled for speed
- `RAG_CONFIG_CONSERVATIVE`: HyDE enabled with single hypothesis for lower latency

**When to use HyDE:**
- ✅ Queries are vague or ambiguous ("auth stuff", "that hash thing")
- ✅ Looking for implementation patterns rather than specific names
- ✅ Natural language queries that don't match code identifiers
- ❌ Query contains exact function/class names
- ❌ Very specific technical queries ("MD5 implementation in file_utils.py")

### Reciprocal Rank Fusion (RRF)

RRF is a proven method for combining multiple ranked result lists without requiring score normalization:

```
RRF_score(d) = 1/(k + rank_semantic) + 1/(k + rank_bm25)
```

Where `k=60` (configurable via `rrf_k` parameter). Benefits:
- **Better Recall**: BM25 can surface documents missed by semantic search
- **No Score Normalization**: Uses ranks, not raw scores
- **Robust**: Handles different score scales across retrieval methods
- **Boosted Consensus**: Documents ranked well by both methods get highest scores

```python
# Adjust RRF constant (default: 60)
# Lower values = more aggressive rank differences
# Higher values = more forgiving of rank differences
results = tool.search_workspace("authentication", rrf_k=60)
```

### Large Repositories

For very large codebases, you may want to:
- Increase `bytes_limit` in config to index larger files
- Adjust `chunk_size` and `chunk_overlap` for your use case
- Use `include_extensions` to limit indexed file types
- Adjust `max_concurrent_requests` and `embedding_batch_size` to index faster

### Performance observations

First cold run might take some time, but does not look drastical on medium hardware/repository:
- Intel i9-11900K / 32 Gb RAM / 3070Ti
- Indexes 882 files (2.1MB total) for 44 sec

The Ollama model itself is the bottleneck, not Python or HTTP overhead.

```
2026-02-06 20:10:06,814 [INFO] ◦ workspace_rag_search_tool ◦ Found 882 files to index (1061 from workspace, 179 binary/non-text skipped)
2026-02-06 20:10:10,007 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 10% (89/882 files, 168.2KB/2.1MB)
2026-02-06 20:10:14,666 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 20% (177/882 files, 392.3KB/2.1MB)
2026-02-06 20:10:22,651 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 30% (265/882 files, 753.2KB/2.1MB)
2026-02-06 20:10:30,602 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 40% (353/882 files, 1.2MB/2.1MB)
2026-02-06 20:10:33,297 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 50% (441/882 files, 1.3MB/2.1MB)
2026-02-06 20:10:34,590 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 60% (530/882 files, 1.4MB/2.1MB)
2026-02-06 20:10:38,650 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 70% (618/882 files, 1.5MB/2.1MB)
2026-02-06 20:10:39,890 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 80% (706/882 files, 1.6MB/2.1MB)
2026-02-06 20:10:43,835 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 90% (794/882 files, 1.8MB/2.1MB)
2026-02-06 20:10:49,135 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 100% (882/882 files, 2.1MB/2.1MB)
2026-02-06 20:10:50,054 [INFO] ◦ workspace_rag_search_tool ◦ Indexed 3152 new chunks (0 files skipped, 0 stale removed)
2026-02-06 20:10:50,055 [INFO] ◦ workspace_rag_search_tool ◦ Workspace index ready!
```