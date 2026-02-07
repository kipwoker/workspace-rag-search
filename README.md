# Workspace RAG Search Tool

A powerful semantic code search tool for local repositories using **RAG (Retrieval-Augmented Generation)** with ChromaDB and local Ollama embeddings. Search your codebase using natural language queries with intelligent hybrid scoring (semantic + keyword matching).

## Features

- 🔍 **Semantic Search** - Find code by meaning, not just keywords
- 🚀 **RRF Hybrid Fusion** - Uses Reciprocal Rank Fusion to combine semantic + BM25 rankings for superior recall
- 🎯 **Cross-Encoder Reranking** - Optional LLM-based reranking for +25-40% relevance improvement
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
> python _examples\example.py _examples\example_in.json
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

2026-02-07 18:35:51,859 [INFO] ◦ workspace_rag_search_tool ◦ Initializing workspace search index for: .../workspace-rag-search
2026-02-07 18:35:52,188 [INFO] ◦ workspace_rag_search_tool ◦ Created new collection: workspace_code_index
2026-02-07 18:35:55,161 [INFO] ◦ workspace_rag_search_tool ◦ Found 35 files to index (36 from workspace, 1 binary/non-text skipped)
2026-02-07 18:35:55,181 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 11% (4/35 files, 629.0B/256.6KB)
2026-02-07 18:35:55,182 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 20% (7/35 files, 9.8KB/256.6KB)
2026-02-07 18:35:55,184 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 31% (11/35 files, 50.2KB/256.6KB)
2026-02-07 18:35:57,445 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 40% (14/35 files, 91.6KB/256.6KB)
2026-02-07 18:35:57,447 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 51% (18/35 files, 112.5KB/256.6KB)
2026-02-07 18:35:57,448 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 60% (21/35 files, 134.9KB/256.6KB)
2026-02-07 18:35:57,450 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 71% (25/35 files, 152.4KB/256.6KB)
2026-02-07 18:35:58,792 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 80% (28/35 files, 185.3KB/256.6KB)
2026-02-07 18:35:58,793 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 91% (32/35 files, 207.2KB/256.6KB)
2026-02-07 18:35:58,794 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 100% (35/35 files, 256.6KB/256.6KB)
2026-02-07 18:36:00,958 [INFO] ◦ reranker.reranker ◦ Initialized CrossEncoderReranker with model=phi3:mini, max_concurrent=5
2026-02-07 18:36:03,248 [INFO] ◦ workspace_rag_search_tool ◦ Reranker initialized with model: phi3:mini
2026-02-07 18:36:03,249 [INFO] ◦ workspace_rag_search_tool ◦ Query cache initialized (max_size=100, ttl=none)
2026-02-07 18:36:03,249 [INFO] ◦ workspace_rag_search_tool ◦ Workspace index ready!
✅ Index ready!

🛠️  tool → → → ◦ [search_workspace] ◦ {"query": "compute file hash defintion", "limit": 3, "path_filter": "utils", "preview_window": 1500}
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
  "results": "Found 3 relevant snippets using RRF + Reranking:

--- Result 1 Final: 0.85 | Rerank: #1 | RRF: 0.0325 | Semantic: #1 | BM25: #2 (semantic: 0.646, bm25: 5.708) ---
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

--- Result 2 Final: 0.04 | Rerank: #2 | RRF: 0.032 | Semantic: #2 | BM25: #3 (semantic: 0.578, bm25: 3.376) ---
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
            return True


--- Result 3 Final: 0.03 | Rerank: #3 | RRF: 0.0323 | Semantic: #3 | BM25: #1 (semantic: 0.526, bm25: 6.108) ---
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
]",
  "query": "compute file hash defintion",
  "reranking": {
    "enabled": true,
    "model": "phi3:mini",
    "latency_ms": 1908.33,
    "candidates": 3
  },
  "metrics": {
    "latency": {
      "semantic_search_ms": 537.21,
      "bm25_build_ms": 20.96,
      "bm25_score_ms": 1.0,
      "rrf_fusion_ms": 0.0,
      "rerank_ms": 1908.33,
      "fetch_results_ms": 1.24,
      "total_ms": 2468.74
    },
    "diversity": {
      "unique_files": 2,
      "file_diversity_ratio": 0.667,
      "score_range": 0.82,
      "score_std": 0.3842,
      "method_agreement": 1.0
    },
    "coverage": {
      "semantic_hits": 3,
      "bm25_hits": 3,
      "both_hits": 3,
      "total_results": 3
    }
  },
  "latency_ms": 2468.74,
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

**Configuration** (in `ollama_config.py`):
```python
from ollama_config import RAGConfig, RERANK_MODEL

RAG_CONFIG_DEFAULT = RAGConfig(
    vector_store_path="./.vectorstore",
    embedding_model="nomic-embed-text",
    chunk_size=1000,
    chunk_overlap=200,
    bytes_limit=100000,
    max_concurrent_requests=10,
    embedding_batch_size=32,
    bm25_implementation="plus",        # BM25 variant: "standard", "plus", "l", "t", "adpt"
    rerank_enabled=True,               # Enable/disable reranking
    rerank_model="phi3:mini",         # Options: "qwen3:0.6b", "phi3:mini"
    rerank_top_k=20,                   # Number of candidates to rerank
    rerank_max_concurrent=5,           # Concurrent requests (lower = less VRAM)
    cache_enabled=True,                # Enable query result caching
    cache_max_size=100,                # Maximum cache entries
    cache_ttl_seconds=None,            # No expiration (cache until refresh)
)
```

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