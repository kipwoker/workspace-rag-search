"""Tests for HyDE query expansion module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from hyde import HyDEQueryExpander, HyDEResult


class TestHyDEResult:
    """Test cases for HyDEResult dataclass."""
    
    def test_hyde_result_creation(self):
        """Test creating a HyDEResult."""
        result = HyDEResult(
            original_query="test query",
            hypothetical_document="This is a test document",
            latency_ms=123.45,
            model_used="qwen3:0.6b"
        )
        
        assert result.original_query == "test query"
        assert result.hypothetical_document == "This is a test document"
        assert result.latency_ms == 123.45
        assert result.model_used == "qwen3:0.6b"


class TestHyDEQueryExpander:
    """Test cases for HyDEQueryExpander."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        expander = HyDEQueryExpander()
        
        assert expander.model_name == "qwen3:0.6b"
        assert expander.ollama_base_url == "http://localhost:11434"
        assert expander.max_tokens == 300
        assert expander.temperature == 0.7
        assert expander.max_concurrent == 3
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        expander = HyDEQueryExpander(
            model_name="phi3:mini",
            ollama_base_url="http://custom:11434",
            max_tokens=500,
            temperature=0.5,
            max_concurrent=5
        )
        
        assert expander.model_name == "phi3:mini"
        assert expander.ollama_base_url == "http://custom:11434"
        assert expander.max_tokens == 500
        assert expander.temperature == 0.5
        assert expander.max_concurrent == 5
    
    def test_init_empty_model_raises(self):
        """Test that empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            HyDEQueryExpander(model_name="")
        
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            HyDEQueryExpander(model_name="   ")
    
    def test_init_invalid_prompt_template(self):
        """Test that prompt template without {query} placeholder raises error."""
        with pytest.raises(ValueError, match="prompt_template must contain {query} placeholder"):
            HyDEQueryExpander(prompt_template="Invalid template without placeholder")
    
    def test_default_prompt_template(self):
        """Test that default prompt template contains {query} placeholder."""
        expander = HyDEQueryExpander()
        assert "{query}" in expander.prompt_template
    
    @patch.object(HyDEQueryExpander, "_generate_async")
    def test_expand_query_success(self, mock_generate):
        """Test successful query expansion."""
        mock_generate.return_value = "Generated hypothetical document"
        
        expander = HyDEQueryExpander()
        result = expander.expand_query("test query")
        
        assert isinstance(result, HyDEResult)
        assert result.original_query == "test query"
        assert result.hypothetical_document == "Generated hypothetical document"
        assert result.model_used == "qwen3:0.6b"
        assert result.latency_ms > 0
        mock_generate.assert_called_once_with("test query")
    
    @patch.object(HyDEQueryExpander, "_generate_async")
    def test_expand_query_empty_response(self, mock_generate):
        """Test query expansion with empty response."""
        mock_generate.return_value = ""
        
        expander = HyDEQueryExpander()
        result = expander.expand_query("test query")
        
        assert result.hypothetical_document == ""
    
    def test_expand_query_empty_query_raises(self):
        """Test that empty query raises ValueError."""
        expander = HyDEQueryExpander()
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            expander.expand_query("")
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            expander.expand_query("   ")
    
    @patch.object(HyDEQueryExpander, "_generate_multi_async")
    def test_expand_query_multi_success(self, mock_generate_multi):
        """Test successful multi-hypothesis expansion."""
        mock_generate_multi.return_value = [
            "Hypothesis 1",
            "Hypothesis 2",
            "Hypothesis 3"
        ]
        
        expander = HyDEQueryExpander()
        results = expander.expand_query_multi("test query", num_hypotheses=3)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, HyDEResult)
            assert result.original_query == "test query"
            assert result.hypothetical_document == f"Hypothesis {i + 1}"
    
    def test_expand_query_multi_invalid_num_hypotheses(self):
        """Test that invalid num_hypotheses raises ValueError."""
        expander = HyDEQueryExpander()
        
        with pytest.raises(ValueError, match="num_hypotheses must be at least 1"):
            expander.expand_query_multi("query", num_hypotheses=0)
        
        with pytest.raises(ValueError, match="num_hypotheses must be at least 1"):
            expander.expand_query_multi("query", num_hypotheses=-1)
    
    def test_add_query_variation(self):
        """Test query variation for diverse hypotheses."""
        expander = HyDEQueryExpander()
        
        base_query = "how to implement auth"
        
        variations = []
        for i in range(6):
            var = expander._add_query_variation(base_query, i)
            variations.append(var)
        
        # All variations should contain the base query
        for var in variations:
            assert "auth" in var.lower() or "implement" in var.lower()
        
        # Variations should be different
        assert len(set(variations)) > 1
    
    @patch("hyde.hyde.httpx.Client")
    def test_check_model_available_success(self, mock_client_class):
        """Test successful model availability check."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:0.6b"},
                {"name": "phi3:mini"}
            ]
        }
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        expander = HyDEQueryExpander(model_name="qwen3:0.6b")
        is_available, error_msg = expander.check_model_available()
        
        assert is_available is True
        assert error_msg is None
    
    @patch("hyde.hyde.httpx.Client")
    def test_check_model_available_not_found(self, mock_client_class):
        """Test model availability check when model not found."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "other-model"}]
        }
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        expander = HyDEQueryExpander(model_name="qwen3:0.6b")
        is_available, error_msg = expander.check_model_available()
        
        assert is_available is False
        assert error_msg is not None
        assert "qwen3:0.6b" in error_msg
    
    @patch("hyde.hyde.httpx.Client")
    def test_check_model_available_base_name_match(self, mock_client_class):
        """Test model availability with base name matching."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "qwen3:latest"}]
        }
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        expander = HyDEQueryExpander(model_name="qwen3:0.6b")
        is_available, error_msg = expander.check_model_available()
        
        # Should match because base name "qwen3" matches
        assert is_available is True
    
    @patch("hyde.hyde.httpx.Client")
    def test_check_model_available_connection_error(self, mock_client_class):
        """Test model availability check with connection error."""
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection refused")
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        expander = HyDEQueryExpander()
        is_available, error_msg = expander.check_model_available()
        
        assert is_available is False
        assert error_msg is not None
        assert "Failed to check" in error_msg
    
    @patch("hyde.hyde.httpx.Client")
    def test_generate_single_sync(self, mock_client_class):
        """Test synchronous document generation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Generated content"}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        expander = HyDEQueryExpander()
        result = expander._generate_sync("test query")
        
        assert result == "Generated content"


class TestHyDEIntegration:
    """Integration-style tests for HyDE."""
    
    def test_alternative_prompt_template(self):
        """Test using the alternative explanation prompt template."""
        from hyde.hyde import HyDEQueryExpander
        
        custom_template = HyDEQueryExpander.EXPLANATION_PROMPT_TEMPLATE
        expander = HyDEQueryExpander(prompt_template=custom_template)
        
        assert expander.prompt_template == custom_template
        assert "{query}" in expander.prompt_template
    
    def test_custom_prompt_template(self):
        """Test using a custom prompt template."""
        custom_template = "Query: {query}\n\nGenerate code example:"
        expander = HyDEQueryExpander(prompt_template=custom_template)
        
        assert expander.prompt_template == custom_template
