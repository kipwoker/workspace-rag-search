"""Maximal Marginal Relevance (MMR) implementation for diversity-aware reranking.

Reference:
    "The Use of MMR, Diversity-Based Reranking for Reordering Documents
     and Producing Summaries" by Carbonell and Goldstein (1998)
    https://www.cs.cmu.edu/~jgc/publication/The_Use_of_MMR_Diversity_Based_Latent_Semantic_Indexing_for_Information_Retrieval.pdf

MMR Formula:
    MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected_docs))

Where:
    - λ (lambda) is the diversity trade-off parameter (0-1)
    - Sim(query, doc) is the relevance score of the document to the query
    - Sim(doc, selected_docs) is the maximum similarity to already selected documents

The algorithm greedily selects documents that balance:
    1. High relevance to the query
    2. Low similarity to already selected documents (diversity)
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity in range [-1, 1]
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def calculate_diversity_penalty(
    doc_embedding: np.ndarray,
    selected_embeddings: List[np.ndarray]
) -> float:
    """Calculate the maximum similarity penalty for diversity.
    
    This represents how similar a document is to the already selected set.
    Higher values mean the document is more similar (less diverse).
    
    Args:
        doc_embedding: Embedding of the candidate document
        selected_embeddings: List of embeddings of already selected documents
        
    Returns:
        Maximum similarity to any selected document (0-1 range)
    """
    if not selected_embeddings:
        return 0.0
    
    max_sim = 0.0
    for selected_emb in selected_embeddings:
        sim = cosine_similarity(doc_embedding, selected_emb)
        # Cosine similarity can be negative, but we care about magnitude
        max_sim = max(max_sim, abs(sim))
    
    return max_sim


class MMRReranker:
    """Maximal Marginal Relevance reranker for diversity-aware result selection.
    
    MMR addresses the problem of result redundancy by explicitly trading off
    relevance against diversity. This is particularly useful for code search
    where multiple chunks from the same file can dominate results.
    
    Key Benefits:
        - Reduces duplicate results from the same files
        - Spreads coverage across different code locations
        - Maintains high relevance through lambda parameter tuning
        - No additional LLM calls (uses existing embeddings)
    
    Example:
        >>> reranker = MMRReranker(lambda_param=0.5)
        >>> 
        >>> # Results from previous search stage (RRF + optional reranking)
        >>> results = [
        ...     ("doc1", 0.95, embedding1, metadata1),
        ...     ("doc2", 0.90, embedding2, metadata2),
        ...     ("doc3", 0.88, embedding3, metadata3),
        ... ]
        >>> 
        >>> # Rerank with diversity consideration
        >>> diverse_results = reranker.rerank(results, top_k=5)
        >>> 
        >>> # Higher lambda = more relevance-focused
        >>> # Lower lambda = more diversity-focused
        >>> reranker_high_diversity = MMRReranker(lambda_param=0.3)
    """
    
    def __init__(
        self,
        lambda_param: float = 0.5,
        max_file_chunks: Optional[int] = None,
        file_penalty_factor: float = 0.1
    ):
        """Initialize the MMR reranker.
        
        Args:
            lambda_param: Trade-off parameter between relevance and diversity (0-1).
                         - 1.0 = Pure relevance (no diversity consideration)
                         - 0.5 = Balanced relevance and diversity (default)
                         - 0.0 = Pure diversity (ignore relevance)
                         Recommended: 0.5-0.7 for code search
            max_file_chunks: Maximum chunks to select from the same file.
                            If None, no file-level limit is enforced.
            file_penalty_factor: Additional penalty factor for selecting multiple
                               chunks from the same file (0-1). Higher values
                               encourage more file diversity.
        
        Raises:
            ValueError: If lambda_param is not in [0, 1]
        """
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError(f"lambda_param must be in [0, 1], got {lambda_param}")
        
        self.lambda_param = lambda_param
        self.max_file_chunks = max_file_chunks
        self.file_penalty_factor = file_penalty_factor
        
        logger.info(
            "Initialized MMRReranker (lambda=%.2f, max_file_chunks=%s, file_penalty=%.2f)",
            lambda_param, max_file_chunks, file_penalty_factor
        )
    
    def rerank(
        self,
        results: List[Tuple[str, float, np.ndarray, Dict]],
        top_k: int = 10
    ) -> List[Tuple[str, float, np.ndarray, Dict]]:
        """Rerank results using Maximal Marginal Relevance.
        
        Args:
            results: List of (doc_id, score, embedding, metadata) tuples.
                    Results should be pre-sorted by relevance score descending.
            top_k: Number of diverse results to select
            
        Returns:
            List of selected (doc_id, mmr_score, embedding, metadata) tuples,
            ordered by selection order (most diverse first)
            
        Example:
            >>> results = [
            ...     ("doc1", 0.95, emb1, {"file_path": "src/auth.py"}),
            ...     ("doc2", 0.90, emb2, {"file_path": "src/auth.py"}),  # Same file
            ...     ("doc3", 0.88, emb3, {"file_path": "src/db.py"}),
            ... ]
            >>> selected = reranker.rerank(results, top_k=2)
            >>> # Likely selects doc1 and doc3 (different files)
        """
        if not results:
            return []
        
        if top_k <= 0:
            return []
        
        # Limit top_k to available results
        top_k = min(top_k, len(results))
        
        # Create working copy with embeddings as numpy arrays
        candidates = []
        for doc_id, score, embedding, metadata in results:
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            candidates.append({
                "doc_id": doc_id,
                "relevance_score": score,
                "embedding": embedding,
                "metadata": metadata
            })
        
        selected: List[Dict] = []
        selected_indices: Set[int] = set()
        file_chunk_counts: Dict[str, int] = {}
        
        while len(selected) < top_k and len(selected_indices) < len(candidates):
            best_mmr_score = float('-inf')
            best_idx = -1
            
            for idx, candidate in enumerate(candidates):
                if idx in selected_indices:
                    continue
                
                # Calculate MMR score
                relevance = candidate["relevance_score"]
                
                # Diversity penalty: max similarity to selected documents
                if selected:
                    selected_embeddings = [s["embedding"] for s in selected]
                    diversity_penalty = calculate_diversity_penalty(
                        candidate["embedding"],
                        selected_embeddings
                    )
                else:
                    diversity_penalty = 0.0
                
                # MMR formula: λ * relevance - (1-λ) * diversity_penalty
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * diversity_penalty
                )
                
                # Apply file-level diversity penalty if configured
                file_path = candidate["metadata"].get("file_path", "")
                if file_path and self.max_file_chunks is not None:
                    file_count = file_chunk_counts.get(file_path, 0)
                    if file_count >= self.max_file_chunks:
                        # Strong penalty for exceeding file limit
                        mmr_score -= 0.5
                    else:
                        # Gradual penalty for multiple chunks from same file
                        file_penalty = file_count * self.file_penalty_factor
                        mmr_score -= file_penalty
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            # Add best candidate to selected
            selected_candidate = candidates[best_idx]
            selected.append({
                **selected_candidate,
                "mmr_score": best_mmr_score,
                "diversity_penalty": calculate_diversity_penalty(
                    selected_candidate["embedding"],
                    [s["embedding"] for s in selected]
                ) if selected else 0.0
            })
            selected_indices.add(best_idx)
            
            # Update file chunk counts
            file_path = selected_candidate["metadata"].get("file_path", "")
            if file_path:
                file_chunk_counts[file_path] = file_chunk_counts.get(file_path, 0) + 1
            
            logger.debug(
                "MMR selected %s (relevance=%.3f, mmr=%.3f, file=%s)",
                selected_candidate["doc_id"],
                selected_candidate["relevance_score"],
                best_mmr_score,
                file_path
            )
        
        # Format output
        output = []
        for item in selected:
            output.append((
                item["doc_id"],
                item["mmr_score"],
                item["embedding"],
                {
                    **item["metadata"],
                    "mmr_relevance_score": item["relevance_score"],
                    "mmr_diversity_penalty": item.get("diversity_penalty", 0.0)
                }
            ))
        
        logger.info(
            "MMR reranking complete: selected %d diverse results from %d candidates",
            len(output), len(candidates)
        )
        
        return output
    
    def rerank_simple(
        self,
        results: List[Tuple[str, float]],
        embeddings: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Simplified MMR reranking for basic use cases.
        
        This version doesn't require metadata and works with just doc_ids,
        scores, and a lookup of embeddings.
        
        Args:
            results: List of (doc_id, score) tuples, sorted by relevance
            embeddings: Dictionary mapping doc_id to embedding vector
            top_k: Number of diverse results to select
            
        Returns:
            List of (doc_id, mmr_score) tuples
        """
        if not results:
            return []
        
        # Convert to full format
        full_results = []
        for doc_id, score in results:
            emb = embeddings.get(doc_id)
            if emb is None:
                logger.warning("No embedding found for %s, skipping", doc_id)
                continue
            full_results.append((doc_id, score, emb, {}))
        
        # Rerank
        reranked = self.rerank(full_results, top_k)
        
        # Return simplified format
        return [(doc_id, mmr_score) for doc_id, mmr_score, _, _ in reranked]


def compute_result_diversity(results: List[Tuple[str, float, np.ndarray, Dict]]) -> Dict:
    """Compute diversity metrics for a set of results.
    
    Args:
        results: List of (doc_id, score, embedding, metadata) tuples
        
    Returns:
        Dictionary with diversity metrics:
            - avg_pairwise_similarity: Average cosine similarity between all pairs
            - max_pairwise_similarity: Maximum similarity between any pair
            - unique_files: Number of unique files represented
            - file_diversity_ratio: Ratio of unique files to total results
    """
    if len(results) < 2:
        return {
            "avg_pairwise_similarity": 0.0,
            "max_pairwise_similarity": 0.0,
            "unique_files": len(results),
            "file_diversity_ratio": 1.0 if results else 0.0
        }
    
    embeddings = [r[2] for r in results]
    
    # Compute pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(abs(sim))
    
    # File diversity
    files = set()
    for r in results:
        metadata = r[3]
        file_path = metadata.get("file_path", "")
        if file_path:
            files.add(file_path)
    
    return {
        "avg_pairwise_similarity": round(sum(similarities) / len(similarities), 4),
        "max_pairwise_similarity": round(max(similarities), 4),
        "unique_files": len(files),
        "file_diversity_ratio": round(len(files) / len(results), 3)
    }
