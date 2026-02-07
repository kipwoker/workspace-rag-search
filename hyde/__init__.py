"""HyDE (Hypothetical Document Embeddings) query expansion module.

This module provides HyDE query expansion for improving retrieval quality
on complex or vague queries by generating hypothetical answer documents.

Reference:
    - HyDE paper: https://arxiv.org/abs/2212.10496
    - Precise Zero-Shot Dense Retrieval without Relevance Labels

Example:
    >>> from hyde import HyDEQueryExpander
    >>> 
    >>> expander = HyDEQueryExpander(
    ...     model_name="qwen3:0.6b",
    ...     ollama_base_url="http://localhost:11434"
    ... )
    >>> 
    >>> # Generate hypothetical document
    >>> query = "how does authentication work"
    >>> hypothetical_doc = expander.generate_hypothetical_document(query)
    >>> print(hypothetical_doc)
    
    >>> # Use for embedding-based search
    >>> expanded_queries = expander.expand_query(query, num_hypotheses=3)
"""

from hyde.hyde import HyDEQueryExpander, HyDEResult

__all__ = ["HyDEQueryExpander", "HyDEResult"]
