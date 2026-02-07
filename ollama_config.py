from dataclasses import dataclass, field
from typing import Literal, Optional

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

# Available reranking models
RerankModel = Literal[
    "phi3:mini",      # Lightweight, fast, good for quick reranking
    "qwen3:0.6b",     # Small but capable, default choice
]

# Default reranking model - uses /api/generate to score relevance
RERANK_MODEL: RerankModel = "phi3:mini"

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
        rerank_enabled: Whether to enable cross-encoder reranking
        rerank_model: Name of the Ollama reranking model to use.
            Options: "phi3:mini", "qwen3:0.6b"
        rerank_top_k: Number of documents to rerank (more = better quality but slower)
        rerank_max_concurrent: Concurrent reranking requests (lower = less VRAM)
        mmr_enabled: Whether to enable MMR diversity reranking
        mmr_lambda: MMR lambda parameter (0-1) for relevance-diversity trade-off.
                   1.0 = pure relevance, 0.0 = pure diversity, 0.5-0.7 recommended
        mmr_max_file_chunks: Maximum chunks to select from the same file.
                            None means no file-level limit.
        mmr_candidates: Number of candidates to consider for MMR reranking.
                       Higher values give more diversity options but slower.
        cache_enabled: Whether to enable query result caching
        cache_max_size: Maximum number of cached query results
        cache_ttl_seconds: Cache entry time-to-live in seconds (None = no expiration)
        metrics_enabled: Whether to enable detailed performance metrics collection
            and display (latency breakdown, diversity scores, coverage metrics)
    
    Example:
        >>> config = RAGConfig(
        ...     vector_store_path="./.vectorstore",
        ...     embedding_model="nomic-embed-text",
        ...     chunk_size=1000,
        ...     chunk_overlap=200,
        ...     bytes_limit=100000,
        ...     max_concurrent_requests=10,
        ...     embedding_batch_size=32,
        ...     bm25_implementation="plus",
        ...     rerank_enabled=True,
        ...     rerank_model="qwen3:0.6b",
        ...     rerank_top_k=20,
        ...     rerank_max_concurrent=5,
        ...     mmr_enabled=True,
        ...     mmr_lambda=0.6,
        ...     mmr_max_file_chunks=2,
        ...     mmr_candidates=20,
        ...     cache_enabled=True,
        ...     cache_max_size=100,
        ...     cache_ttl_seconds=300,
        ...     metrics_enabled=True
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
    rerank_enabled: bool = field(default=True)
    rerank_model: RerankModel = field(default=RERANK_MODEL)
    rerank_top_k: int = field(default=20)
    rerank_max_concurrent: int = field(default=5)
    mmr_enabled: bool = field(default=True)
    mmr_lambda: float = field(default=0.6)
    mmr_max_file_chunks: Optional[int] = field(default=2)
    mmr_candidates: int = field(default=20)
    cache_enabled: bool = field(default=True)
    cache_max_size: int = field(default=100)
    cache_ttl_seconds: Optional[int] = field(default=None)
    metrics_enabled: bool = field(default=True)

RAG_CONFIG_DEFAULT = RAGConfig(
    vector_store_path="./.vectorstore",
    embedding_model=MODELS["embeddings"]["name"],
    chunk_size=1000,
    chunk_overlap=200,
    bytes_limit=100000,
    max_concurrent_requests=10,  # 10 concurrent embedding requests
    embedding_batch_size=32,     # 32 texts per request
    bm25_implementation="plus",  # Default to BM25+ for better RRF fusion
    rerank_enabled=True,         # Enable reranking by default
    rerank_model=RERANK_MODEL,
    rerank_top_k=20,             # Rerank top 20 RRF results
    rerank_max_concurrent=5,     # 5 concurrent reranking requests
    mmr_enabled=True,            # Enable MMR for better diversity
    mmr_lambda=0.6,              # Higher lambda for more relevance focus
    mmr_max_file_chunks=2,       # Only 1 chunk per file for maximum spread
    mmr_candidates=20,           # Number of candidates to consider
    cache_enabled=True,          # Enable query caching by default
    cache_max_size=100,          # Cache up to 100 queries
    cache_ttl_seconds=None,      # No expiration (cache until index refresh)
    metrics_enabled=True,        # Enable performance metrics by default
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
    rerank_enabled=False,        # Disable reranking for speed
    rerank_model=RERANK_MODEL,
    rerank_top_k=20,
    rerank_max_concurrent=5,
    mmr_enabled=False,           # Disable MMR for speed
    cache_enabled=True,          # Enable caching for repeated queries
    cache_max_size=50,           # Smaller cache for memory efficiency
    cache_ttl_seconds=60,        # 1 minute TTL for fast-changing code
    metrics_enabled=False,       # Disable metrics for minimal overhead
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
    rerank_enabled=True,
    rerank_model=RERANK_MODEL,
    rerank_top_k=10,             # Rerank fewer documents
    rerank_max_concurrent=2,     # Lower VRAM usage
    mmr_enabled=True,            # Enable MMR for better diversity
    mmr_lambda=0.7,              # Higher lambda for more relevance focus
    mmr_max_file_chunks=1,       # Only 1 chunk per file for maximum spread
    mmr_candidates=15,
    cache_enabled=True,          # Enable caching
    cache_max_size=200,          # Larger cache since we're conservative on speed
    cache_ttl_seconds=600,       # 10 minute TTL
    metrics_enabled=True,        # Enable metrics for observability
)
