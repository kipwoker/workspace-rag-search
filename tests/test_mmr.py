"""Tests for Maximal Marginal Relevance (MMR) diversity reranking."""

import numpy as np
import pytest

from mmr import MMRReranker, calculate_diversity_penalty, compute_result_diversity


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""
    
    def test_identical_vectors(self):
        """Cosine similarity of identical vectors should be 1."""
        from mmr.mmr import cosine_similarity
        vec = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-6)
    
    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors should be 0."""
        from mmr.mmr import cosine_similarity
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0, abs=1e-6)
    
    def test_opposite_vectors(self):
        """Cosine similarity of opposite vectors should be -1."""
        from mmr.mmr import cosine_similarity
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0, abs=1e-6)
    
    def test_zero_vector(self):
        """Cosine similarity with zero vector should be 0."""
        from mmr.mmr import cosine_similarity
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(vec1, vec2) == 0.0


class TestDiversityPenalty:
    """Tests for diversity penalty calculation."""
    
    def test_no_selected_documents(self):
        """Diversity penalty should be 0 when no documents selected."""
        vec = np.array([1.0, 2.0, 3.0])
        penalty = calculate_diversity_penalty(vec, [])
        assert penalty == 0.0
    
    def test_similar_documents_high_penalty(self):
        """Similar documents should have high diversity penalty."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.99, 0.01, 0.0])  # Very similar to vec1
        
        penalty = calculate_diversity_penalty(vec1, [vec2])
        assert penalty > 0.9  # High similarity
    
    def test_dissimilar_documents_low_penalty(self):
        """Dissimilar documents should have low diversity penalty."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])  # Orthogonal to vec1
        
        penalty = calculate_diversity_penalty(vec1, [vec2])
        assert penalty < 0.1  # Low similarity
    
    def test_max_similarity_used(self):
        """Maximum similarity to any selected document should be used."""
        vec = np.array([1.0, 0.0, 0.0])
        selected = [
            np.array([0.0, 1.0, 0.0]),  # Orthogonal (sim = 0)
            np.array([0.99, 0.01, 0.0]),  # Very similar (sim ≈ 0.99)
        ]
        
        penalty = calculate_diversity_penalty(vec, selected)
        assert penalty > 0.9  # Should use max, which is the similar one


class TestMMRReranker:
    """Tests for MMRReranker class."""
    
    def test_lambda_validation(self):
        """Invalid lambda values should raise ValueError."""
        with pytest.raises(ValueError):
            MMRReranker(lambda_param=-0.1)
        
        with pytest.raises(ValueError):
            MMRReranker(lambda_param=1.1)
        
        # Valid values should not raise
        MMRReranker(lambda_param=0.0)
        MMRReranker(lambda_param=0.5)
        MMRReranker(lambda_param=1.0)
    
    def test_empty_results(self):
        """Empty results should return empty list."""
        reranker = MMRReranker(lambda_param=0.5)
        results = reranker.rerank([], top_k=5)
        assert results == []
    
    def test_top_k_zero(self):
        """top_k=0 should return empty list."""
        reranker = MMRReranker(lambda_param=0.5)
        results = [
            ("doc1", 0.9, np.array([1.0, 0.0]), {}),
        ]
        assert reranker.rerank(results, top_k=0) == []
    
    def test_pure_relevance_ranking(self):
        """lambda=1 should rank purely by relevance."""
        reranker = MMRReranker(lambda_param=1.0)
        
        results = [
            ("doc1", 0.9, np.array([1.0, 0.0]), {"file_path": "a.py"}),
            ("doc2", 0.8, np.array([0.99, 0.01]), {"file_path": "b.py"}),  # Similar to doc1
            ("doc3", 0.7, np.array([0.0, 1.0]), {"file_path": "c.py"}),
        ]
        
        reranked = reranker.rerank(results, top_k=3)
        
        # With pure relevance, order should be preserved
        assert [r[0] for r in reranked] == ["doc1", "doc2", "doc3"]
    
    def test_pure_diversity_ranking(self):
        """lambda=0 should rank purely by diversity."""
        reranker = MMRReranker(lambda_param=0.0)
        
        # Create documents where doc1 and doc2 are similar, doc3 is different
        results = [
            ("doc1", 0.9, np.array([1.0, 0.0]), {"file_path": "a.py"}),
            ("doc2", 0.8, np.array([0.99, 0.01]), {"file_path": "b.py"}),  # Similar to doc1
            ("doc3", 0.7, np.array([0.0, 1.0]), {"file_path": "c.py"}),  # Different
        ]
        
        reranked = reranker.rerank(results, top_k=3)
        
        # With pure diversity, doc3 should come second (most different from doc1)
        doc_ids = [r[0] for r in reranked]
        assert doc_ids[0] == "doc1"  # First is always highest relevance
        assert doc_ids[2] in ["doc1", "doc2"]  # Most similar should be last
    
    def test_file_diversity_limit(self):
        """max_file_chunks should limit chunks from same file."""
        # Use a strong file penalty to ensure diversity
        reranker = MMRReranker(lambda_param=0.5, max_file_chunks=1, file_penalty_factor=0.5)
        
        results = [
            ("doc1", 0.9, np.array([1.0, 0.0]), {"file_path": "same.py"}),
            ("doc2", 0.8, np.array([0.99, 0.01]), {"file_path": "same.py"}),  # Same file
            ("doc3", 0.7, np.array([0.0, 1.0]), {"file_path": "other.py"}),  # Different file
        ]
        
        reranked = reranker.rerank(results, top_k=3)
        doc_ids = [r[0] for r in reranked]
        
        # With strong penalty, doc3 should be selected before doc2 (same file as doc1)
        assert "doc3" in doc_ids[:2]  # doc3 should be in top 2
    
    def test_top_k_limits_results(self):
        """top_k should limit the number of results."""
        reranker = MMRReranker(lambda_param=0.5)
        
        results = [
            (f"doc{i}", 0.9 - i * 0.01, np.array([float(i), 0.0]), {"file_path": f"{i}.py"})
            for i in range(10)
        ]
        
        reranked = reranker.rerank(results, top_k=5)
        assert len(reranked) == 5
    
    def test_mmr_score_calculation(self):
        """MMR scores should combine relevance and diversity."""
        reranker = MMRReranker(lambda_param=0.5)
        
        results = [
            ("doc1", 1.0, np.array([1.0, 0.0]), {}),
            ("doc2", 0.5, np.array([1.0, 0.0]), {}),  # Same embedding, lower relevance
        ]
        
        reranked = reranker.rerank(results, top_k=2)
        
        # doc2 should have lower MMR score due to similarity to doc1
        mmr_scores = {r[0]: r[1] for r in reranked}
        assert mmr_scores["doc1"] > mmr_scores["doc2"]


class TestComputeResultDiversity:
    """Tests for diversity metrics computation."""
    
    def test_empty_results(self):
        """Empty results should return zero metrics."""
        metrics = compute_result_diversity([])
        assert metrics["unique_files"] == 0
        assert metrics["file_diversity_ratio"] == 0.0
    
    def test_single_result(self):
        """Single result should have perfect diversity ratio."""
        results = [
            ("doc1", 1.0, np.array([1.0, 0.0]), {"file_path": "a.py"}),
        ]
        metrics = compute_result_diversity(results)
        assert metrics["unique_files"] == 1
        assert metrics["file_diversity_ratio"] == 1.0
    
    def test_identical_embeddings(self):
        """Identical embeddings should have high similarity."""
        emb = np.array([1.0, 0.0])
        results = [
            ("doc1", 1.0, emb, {"file_path": "a.py"}),
            ("doc2", 0.9, emb, {"file_path": "b.py"}),
        ]
        metrics = compute_result_diversity(results)
        assert metrics["avg_pairwise_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["max_pairwise_similarity"] == pytest.approx(1.0, abs=1e-6)
    
    def test_diverse_embeddings(self):
        """Orthogonal embeddings should have low similarity."""
        results = [
            ("doc1", 1.0, np.array([1.0, 0.0]), {"file_path": "a.py"}),
            ("doc2", 0.9, np.array([0.0, 1.0]), {"file_path": "b.py"}),
        ]
        metrics = compute_result_diversity(results)
        assert metrics["avg_pairwise_similarity"] < 0.1
        assert metrics["max_pairwise_similarity"] < 0.1
    
    def test_file_diversity_calculation(self):
        """File diversity should count unique files."""
        results = [
            ("doc1", 1.0, np.array([1.0, 0.0]), {"file_path": "a.py"}),
            ("doc2", 0.9, np.array([0.0, 1.0]), {"file_path": "a.py"}),  # Same file
            ("doc3", 0.8, np.array([0.5, 0.5]), {"file_path": "b.py"}),  # Different file
        ]
        metrics = compute_result_diversity(results)
        assert metrics["unique_files"] == 2
        # Allow for rounding (computed value is rounded to 3 decimal places)
        assert metrics["file_diversity_ratio"] == pytest.approx(2/3, abs=0.01)


class TestRerankSimple:
    """Tests for the simplified reranking interface."""
    
    def test_basic_functionality(self):
        """rerank_simple should work with minimal input."""
        reranker = MMRReranker(lambda_param=0.5)
        
        results = [("doc1", 0.9), ("doc2", 0.8)]
        embeddings = {
            "doc1": np.array([1.0, 0.0]),
            "doc2": np.array([0.0, 1.0]),
        }
        
        reranked = reranker.rerank_simple(results, embeddings, top_k=2)
        
        assert len(reranked) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in reranked)
    
    def test_missing_embedding(self):
        """Documents without embeddings should be skipped."""
        reranker = MMRReranker(lambda_param=0.5)
        
        results = [("doc1", 0.9), ("doc2", 0.8)]
        embeddings = {
            "doc1": np.array([1.0, 0.0]),
            # doc2 embedding missing
        }
        
        reranked = reranker.rerank_simple(results, embeddings, top_k=2)
        
        assert len(reranked) == 1
        assert reranked[0][0] == "doc1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
