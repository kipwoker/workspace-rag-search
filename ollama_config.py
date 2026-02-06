from dataclasses import dataclass, field
from typing import Literal

OLLAMA_BASE_URL = "http://localhost:11434"

MODELS = {
    "embeddings": {
        "name": "nomic-embed-text",
    },
}

# Available BM25 implementations
BM25Implementation = Literal[
    "standard",   # Standard BM25 (bm25_scorer)
    "plus",       # BM25+ with lower bound for zero-frequency terms
    "l",          # BM25L with logarithmic TF normalization
    "t",          # BM25T with two-stage TF transformation
    "adpt",       # BM25-Adpt with adaptive parameter tuning
]

@dataclass
class RAGConfig:
    """RAG configuration model.
    
    Attributes:
        vector_store_path: Path to store the vector database
        embedding_model: Name of the Ollama embedding model to use
        chunk_size: Size of text chunks for indexing
        chunk_overlap: Overlap between chunks for better context
        bytes_limit: Maximum file size to index (in bytes)
        max_concurrent_requests: Number of concurrent embedding requests
        embedding_batch_size: Number of texts per embedding API call
        bm25_implementation: BM25 variant to use for keyword scoring
            Options: "standard", "plus", "l", "t", "adpt"
    
    Example:
        >>> config = RAGConfig(
        ...     vector_store_path="./.vectorstore",
        ...     embedding_model="nomic-embed-text",
        ...     chunk_size=1000,
        ...     chunk_overlap=200,
        ...     bytes_limit=100000,
        ...     max_concurrent_requests=10,
        ...     embedding_batch_size=32,
        ...     bm25_implementation="plus"
        ... )
    """
    vector_store_path: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    bytes_limit: int
    max_concurrent_requests: int = field(default=10)
    embedding_batch_size: int = field(default=32)
    bm25_implementation: BM25Implementation = field(default="plus")

RAG_CONFIG_DEFAULT = RAGConfig(
    vector_store_path="./.vectorstore",
    embedding_model=MODELS["embeddings"]["name"],
    chunk_size=1000,
    chunk_overlap=200,
    bytes_limit=100000,
    max_concurrent_requests=10,  # 10 concurrent embedding requests
    embedding_batch_size=32,     # 32 texts per request
    bm25_implementation="plus",  # Default to BM25+ for better RRF fusion
)


# Performance presets for different use cases
RAG_CONFIG_FAST = RAGConfig(
    vector_store_path="./.vectorstore",
    embedding_model=MODELS["embeddings"]["name"],
    chunk_size=1000,
    chunk_overlap=200,
    bytes_limit=100000,
    max_concurrent_requests=20,  # Higher concurrency for faster indexing
    embedding_batch_size=64,     # Larger batches for fewer API calls
    bm25_implementation="plus",
)

RAG_CONFIG_CONSERVATIVE = RAGConfig(
    vector_store_path="./.vectorstore",
    embedding_model=MODELS["embeddings"]["name"],
    chunk_size=1000,
    chunk_overlap=200,
    bytes_limit=100000,
    max_concurrent_requests=5,   # Lower concurrency for limited resources
    embedding_batch_size=16,     # Smaller batches
    bm25_implementation="plus",
)
