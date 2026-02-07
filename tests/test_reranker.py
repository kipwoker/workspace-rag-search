"""Tests for cross-encoder reranker module."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch, AsyncMock
import httpx

from reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """Test cases for CrossEncoderReranker."""

    def test_init_default(self):
        """Test default initialization."""
        reranker = CrossEncoderReranker()
        
        assert reranker.model_name == "qwen3:0.6b"
        assert reranker.ollama_base_url == "http://localhost:11434"
        assert reranker.max_concurrent == 5
        assert reranker.timeout == 60.0
        assert reranker.max_sequence_length is None
    
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        reranker = CrossEncoderReranker(
            model_name="phi3:mini",
            ollama_base_url="http://ollama:11434",
            max_concurrent=3,
            timeout=30.0,
            max_sequence_length=512
        )
        
        assert reranker.model_name == "phi3:mini"
        assert reranker.ollama_base_url == "http://ollama:11434"
        assert reranker.max_concurrent == 3
        assert reranker.timeout == 30.0
        assert reranker.max_sequence_length == 512
    
    def test_init_empty_model_raises(self):
        """Test that empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            CrossEncoderReranker(model_name="")
        
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            CrossEncoderReranker(model_name="   ")
    
    @patch("httpx.Client")
    def test_check_model_available_success(self, mock_client_class):
        """Test model availability check when model exists."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:0.6b"},
                {"name": "nomic-embed-text:latest"}
            ]
        }
        mock_response.raise_for_status = Mock()
        
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__ = Mock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = Mock(return_value=False)
        
        reranker = CrossEncoderReranker()
        is_available, error_msg = reranker.check_model_available()
        
        assert is_available is True
        assert error_msg is None
    
    @patch("httpx.Client")
    def test_check_model_available_not_found(self, mock_client_class):
        """Test model availability check when model doesn't exist."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "nomic-embed-text:latest"}
            ]
        }
        mock_response.raise_for_status = Mock()
        
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__ = Mock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = Mock(return_value=False)
        
        reranker = CrossEncoderReranker(model_name="nonexistent-model:latest")
        is_available, error_msg = reranker.check_model_available()
        
        assert is_available is False
        assert "not found" in error_msg
    
    @patch("httpx.Client")
    def test_check_model_available_connection_error(self, mock_client_class):
        """Test model availability check with connection error."""
        mock_client = Mock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client_class.return_value.__enter__ = Mock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = Mock(return_value=False)
        
        reranker = CrossEncoderReranker()
        is_available, error_msg = reranker.check_model_available()
        
        assert is_available is False
        assert "Failed to check" in error_msg


class TestScoreParsing:
    """Test cases for score parsing."""

    def test_parse_score_number(self):
        """Test parsing plain numbers."""
        reranker = CrossEncoderReranker()
        
        assert reranker._parse_score("85") == 0.85
        assert reranker._parse_score("100") == 1.0
        assert reranker._parse_score("0") == 0.0
        assert reranker._parse_score("50") == 0.5
    
    def test_parse_score_decimal(self):
        """Test parsing decimal scores."""
        reranker = CrossEncoderReranker()
        
        assert reranker._parse_score("85.5") == 0.855
        assert reranker._parse_score("0.75") == 0.75
        assert reranker._parse_score("0.95") == 0.95
    
    def test_parse_score_with_text(self):
        """Test parsing scores embedded in text."""
        reranker = CrossEncoderReranker()
        
        assert reranker._parse_score("The score is 85") == 0.85
        assert reranker._parse_score("Score: 90") == 0.9
        assert reranker._parse_score("Relevance: 75/100") == 0.75
    
    def test_parse_score_already_normalized(self):
        """Test parsing already normalized scores."""
        reranker = CrossEncoderReranker()
        
        assert reranker._parse_score("0.85") == 0.85
        assert reranker._parse_score("1.0") == 1.0
        assert reranker._parse_score("0.0") == 0.0
    
    def test_parse_score_empty(self):
        """Test parsing empty response returns default fallback score."""
        reranker = CrossEncoderReranker()
        
        # Empty or non-numeric responses return default 0.5
        assert reranker._parse_score("") == 0.5
        assert reranker._parse_score("   ") == 0.5
        assert reranker._parse_score("no number here") == 0.5
        assert reranker._parse_score("just text") == 0.5


class TestPromptFormatting:
    """Test cases for prompt formatting."""

    def test_format_prompt_default(self):
        """Test default prompt formatting."""
        reranker = CrossEncoderReranker()
        
        prompt = reranker._format_prompt("test query", "test document")
        
        assert "Query: test query" in prompt
        assert "Document: test document" in prompt
        assert "how relevant" in prompt.lower()
    
    def test_format_prompt_custom_template(self):
        """Test custom prompt template."""
        template = "Q: {query}\nD: {document}\nScore:"
        reranker = CrossEncoderReranker(prompt_template=template)
        
        prompt = reranker._format_prompt("test query", "test document")
        
        assert "Q: test query" in prompt
        assert "D: test document" in prompt
        assert "Score:" in prompt
    
    def test_format_prompt_truncation(self):
        """Test document truncation when max_sequence_length is set."""
        reranker = CrossEncoderReranker(max_sequence_length=10)  # Very small for testing
        
        long_doc = "A" * 1000
        prompt = reranker._format_prompt("query", long_doc)
        
        # Document should be truncated
        assert "..." in prompt or len(prompt) < 1100


class TestRerank:
    """Test cases for the rerank method."""

    @patch("reranker.reranker.CrossEncoderReranker._rerank_async")
    def test_rerank_empty_documents(self, mock_rerank_async):
        """Test reranking with empty document list."""
        reranker = CrossEncoderReranker()
        
        results = reranker.rerank("query", [])
        
        assert results == []
        mock_rerank_async.assert_not_called()
    
    @patch("reranker.reranker.CrossEncoderReranker._rerank_async")
    def test_rerank_empty_query(self, mock_rerank_async):
        """Test reranking with empty query."""
        reranker = CrossEncoderReranker()
        
        docs = [("doc1", "content1"), ("doc2", "content2")]
        results = reranker.rerank("", docs, top_k=2)
        
        # Should return documents with default scores
        assert len(results) == 2
        assert all(score == 0.5 for _, score, _ in results)
        mock_rerank_async.assert_not_called()
    
    @patch("reranker.reranker.CrossEncoderReranker._rerank_async")
    def test_rerank_sorts_by_score(self, mock_rerank_async):
        """Test that results are sorted by score descending."""
        mock_rerank_async.return_value = [0.3, 0.9, 0.5]
        
        reranker = CrossEncoderReranker()
        docs = [("doc1", "content1"), ("doc2", "content2"), ("doc3", "content3")]
        
        results = reranker.rerank("query", docs, top_k=3)
        
        # Should be sorted: doc2 (0.9), doc3 (0.5), doc1 (0.3)
        assert results[0][0] == "doc2"
        assert results[0][1] == 0.9
        assert results[1][0] == "doc3"
        assert results[1][1] == 0.5
        assert results[2][0] == "doc1"
        assert results[2][1] == 0.3
    
    @patch("reranker.reranker.CrossEncoderReranker._rerank_async")
    def test_rerank_respects_top_k(self, mock_rerank_async):
        """Test that top_k limits the number of results."""
        mock_rerank_async.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        reranker = CrossEncoderReranker()
        docs = [(f"doc{i}", f"content{i}") for i in range(5)]
        
        results = reranker.rerank("query", docs, top_k=3)
        
        assert len(results) == 3
        assert results[0][0] == "doc0"
        assert results[2][0] == "doc2"
    
    @patch("reranker.reranker.CrossEncoderReranker._rerank_async")
    def test_rerank_handles_errors(self, mock_rerank_async):
        """Test that errors in scoring return 0.0 for failed documents."""
        mock_rerank_async.return_value = [0.9, Exception("API error"), 0.5]
        
        reranker = CrossEncoderReranker()
        docs = [("doc1", "content1"), ("doc2", "content2"), ("doc3", "content3")]
        
        results = reranker.rerank("query", docs, top_k=3)
        
        # Error document should have 0.0 score
        assert len(results) == 3
        assert results[2][0] == "doc2"  # Lowest due to 0.0
        assert results[2][1] == 0.0


class TestAsyncReranking:
    """Test cases for async reranking functionality."""

    @pytest.mark.asyncio
    @patch("reranker.reranker.CrossEncoderReranker._score_pair_async")
    async def test_rerank_async_success(self, mock_score_pair):
        """Test successful async reranking."""
        mock_score_pair.return_value = 0.85
        
        reranker = CrossEncoderReranker(max_concurrent=2)
        docs = [("doc1", "content1"), ("doc2", "content2")]
        
        scores = await reranker._rerank_async("query", docs)
        
        assert len(scores) == 2
        assert all(s == 0.85 for s in scores)
        assert mock_score_pair.call_count == 2
