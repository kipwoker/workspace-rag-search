"""Reciprocal Rank Fusion (RRF) implementation.

Reference:
    "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
"""

from typing import Dict, List, Tuple


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists.

    RRF fuses rankings from multiple retrieval methods using the formula:
        RRF_score(d) = sum(1 / (k + rank_d))

    Where:
        - k is a constant (typically 60) that dampens the impact of low rankings
        - rank_d is the document's rank in each result list

    Benefits:
        - No score normalization needed (uses ranks)
        - Robust to different score scales across methods
        - Documents appearing in multiple lists get boosted
        - Simple yet effective fusion approach

    Example:
        >>> rrf = ReciprocalRankFusion(k=60)
        >>> semantic_results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        >>> bm25_results = [("doc2", 0.95), ("doc4", 0.85), ("doc1", 0.75)]
        >>> fused = rrf.fuse([semantic_results, bm25_results], limit=3)
    """

    def __init__(self, k: int = 60):
        """Initialize RRF with tuning parameter k.

        Args:
            k: RRF constant (default 60). Higher values reduce the impact of ranking differences.
               k=60 is the standard value from the original paper.
        """
        self.k = k

    def fuse(
        self,
        ranked_lists: List[List[Tuple[str, float]]],
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """Fuse multiple ranked lists using RRF.

        Args:
            ranked_lists: List of ranked result lists, where each list contains
                         (doc_id, score) tuples sorted by relevance (best first)
            limit: Maximum number of results to return

        Returns:
            List of (doc_id, rrf_score) tuples sorted by RRF score descending
        """
        rrf_scores: Dict[str, float] = {}

        for ranked_list in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked_list, start=1):
                # RRF formula: score = 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]
