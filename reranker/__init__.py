"""Cross-encoder reranking for improving search result relevance.

This module provides cross-encoder based reranking using Ollama models
to significantly improve the quality of search results by computing
query-document relevance scores.

Cross-encoders are more accurate than bi-encoders (embeddings) because
they can attend to both query and document simultaneously, capturing
fine-grained semantic relationships.

Example:
    >>> from reranker import CrossEncoderReranker
    >>> reranker = CrossEncoderReranker(
    ...     model_name="your-reranker-model",
    ...     ollama_base_url="http://localhost:11434"
    ... )
    >>> 
    >>> # Rerank documents
    >>> query = "authentication middleware"
    >>> documents = [("doc1", "user login code"), ("doc2", "jwt token handling")]
    >>> reranked = reranker.rerank(query, documents, top_k=5)
"""

from reranker.reranker import CrossEncoderReranker

__all__ = ["CrossEncoderReranker"]
