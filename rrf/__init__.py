"""Reciprocal Rank Fusion (RRF) implementation for combining multiple ranked result lists.

This module provides the ReciprocalRankFusion class for fusing rankings from
multiple retrieval methods using the RRF formula.

Example:
    >>> from rrf import ReciprocalRankFusion
    >>> rrf = ReciprocalRankFusion(k=60)
    >>> semantic_results = [("doc1", 0.9), ("doc2", 0.8)]
    >>> bm25_results = [("doc2", 0.95), ("doc3", 0.7)]
    >>> fused = rrf.fuse([semantic_results, bm25_results], limit=3)
"""

from .rrf import ReciprocalRankFusion

__all__ = ["ReciprocalRankFusion"]
