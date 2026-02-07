"""Maximal Marginal Relevance (MMR) diversity reranking.

This module provides MMR-based diversity reranking to reduce duplicate
results from the same files while maintaining query relevance.
"""

from .mmr import MMRReranker, calculate_diversity_penalty, compute_result_diversity

__all__ = ["MMRReranker", "calculate_diversity_penalty", "compute_result_diversity"]
