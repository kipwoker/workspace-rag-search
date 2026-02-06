from dataclasses import dataclass, field

OLLAMA_BASE_URL = "http://localhost:11434"

MODELS = {
    "embeddings": {
        "name": "nomic-embed-text",
    },
}

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
    
    Example:
        >>> config = RAGConfig(
        ...     vector_store_path="./.vectorstore",
        ...     embedding_model="nomic-embed-text",
        ...     chunk_size=1000,
        ...     chunk_overlap=200,
        ...     bytes_limit=100000,
        ...     max_concurrent_requests=10,
        ...     embedding_batch_size=32
        ... )
    """
    vector_store_path: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    bytes_limit: int
    max_concurrent_requests: int = field(default=10)
    embedding_batch_size: int = field(default=32)

RAG_CONFIG_DEFAULT = RAGConfig(
    vector_store_path="./.vectorstore",
    embedding_model=MODELS["embeddings"]["name"],
    chunk_size=1000,
    chunk_overlap=200,
    bytes_limit=100000,
    max_concurrent_requests=10,  # 10 concurrent embedding requests
    embedding_batch_size=32,     # 32 texts per request
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
)

RAG_CONFIG_CONSERVATIVE = RAGConfig(
    vector_store_path="./.vectorstore",
    embedding_model=MODELS["embeddings"]["name"],
    chunk_size=1000,
    chunk_overlap=200,
    bytes_limit=100000,
    max_concurrent_requests=5,   # Lower concurrency for limited resources
    embedding_batch_size=16,     # Smaller batches
)
