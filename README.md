# Workspace RAG Search Tool

A powerful semantic code search tool for local repositories using **RAG (Retrieval-Augmented Generation)** with ChromaDB and local Ollama embeddings. Search your codebase using natural language queries with intelligent hybrid scoring (semantic + keyword matching).

## Features

- 🔍 **Semantic Search** - Find code by meaning, not just keywords
- 🚀 **Hybrid Scoring** - Combines semantic similarity (60%) with keyword matching (40%)
- 📁 **Smart Indexing** - Incremental updates, only processes changed files
- 🦙 **Local Embeddings** - Uses Ollama for 100% local, private embeddings
- 🚫 **Git-Aware** - Automatically respects `.gitignore` rules
- 🧠 **Content Detection** - Automatically skips binary files
- ⚡ **Fast** - Persistent vector store with ChromaDB
- 🎯 **Configurable** - Customizable chunking, extensions, and filters

## Prerequisites

1. **Python 3.9+**
2. **Ollama**

### Install Model

```
ollama pull nomic-embed-text
```
- `nomic-embed-text` is default model (configure your own in `ollama_config.py`)

## Installation

```
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

2026-02-06 20:48:58,461 [INFO] ◦ workspace_rag_search_tool ◦ Initializing workspace search index for: .../workspace_rag_search_tool
2026-02-06 20:48:58,742 [INFO] ◦ workspace_rag_search_tool ◦ Created new collection: workspace_code_index
2026-02-06 20:49:01,492 [INFO] ◦ workspace_rag_search_tool ◦ Found 12 files to index (13 from workspace, 1 binary/non-text skipped)
2026-02-06 20:49:01,514 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 16% (2/12 files, 6.5KB/60.2KB)
2026-02-06 20:49:01,515 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 25% (3/12 files, 13.1KB/60.2KB)
2026-02-06 20:49:01,515 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 33% (4/12 files, 15.4KB/60.2KB)
2026-02-06 20:49:01,515 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 41% (5/12 files, 17.2KB/60.2KB)
2026-02-06 20:49:01,517 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 50% (6/12 files, 26.7KB/60.2KB)
2026-02-06 20:49:01,518 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 66% (8/12 files, 27.4KB/60.2KB)
2026-02-06 20:49:01,518 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 75% (9/12 files, 29.4KB/60.2KB)
2026-02-06 20:49:01,518 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 83% (10/12 files, 33.2KB/60.2KB)
2026-02-06 20:49:01,519 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 91% (11/12 files, 35.1KB/60.2KB)
2026-02-06 20:49:01,519 [INFO] ◦ workspace_rag_search_tool ◦ Indexing progress: 100% (12/12 files, 60.2KB/60.2KB)
2026-02-06 20:49:03,019 [INFO] ◦ workspace_rag_search_tool ◦ Indexed 81 new chunks (0 files skipped, 0 stale removed)
2026-02-06 20:49:03,019 [INFO] ◦ workspace_rag_search_tool ◦ Workspace index ready!
✅ Index ready!

🛠️ tool → → → ◦ [search_workspace] ◦ {"query": "compute file hash defintion", "limit": 3, "path_filter": "utils", "preview_window": 1500}
📄 tool ← ← ← ◦ [search_workspace] ◦
{
  "status": "success",
  "count": 3,
  "total_candidates": 7,
  "results": "Found 7 relevant snippets, showing top 3: matching 'utils'

--- Result 1 (semantic: 0.647, keyword: 0.75, blended: 0.688) ---
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
        Human readable string (e....

--- Result 2 (semantic: 0.53, keyword: 0.75, blended: 0.618) ---
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
]...

--- Result 3 (semantic: 0.577, keyword: 0.5, blended: 0.546) ---
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
  ...",
  "query": "compute file hash defintion"
}
```
</details>

## Configuration

Edit `ollama_config.py` to customize behavior.

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
  "count": 5,
  "total_candidates": 23,
  "results": "Found 23 relevant snippets, showing top 5:\n\n--- Result 1 (semantic: 0.92, keyword: 0.8, blended: 0.87) ---\n[File: src/auth.py]\n\ndef authenticate_user(token):\n    ...",
  "query": "authentication middleware"
}
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
   - Retrieves semantically similar chunks
   - Applies hybrid scoring (semantic + keyword)
   - Returns ranked results with file context

> [!WARNING]
> The `preview_window` parameter limits how many characters are displayed from the start of each result. If your search term appears later in the chunk, it may not be visible in the truncated preview. Set `preview_window=None` (default) to display the full chunk content and ensure matches are always visible.


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