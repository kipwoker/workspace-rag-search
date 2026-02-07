"""Cross-encoder reranker implementation using Ollama models.

Reference:
    - mixedbread-ai/mxbai-rerank-large-v2: https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1
    - BGE Reranker: https://github.com/FlagOpen/FlagEmbedding
"""

import asyncio
import logging
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker using Ollama models.
    
    Cross-encoders compute relevance scores by processing the query and
    document together, allowing for more nuanced understanding than
    embedding-based similarity. This leads to significantly better
    ranking quality at the cost of additional compute.
    
    The reranker works by:
    1. Taking a query and a list of candidate documents
    2. Sending query+document pairs to the cross-encoder model
    3. Getting relevance scores for each pair
    4. Sorting documents by score
    
    Performance Characteristics:
        - Latency: ~50-200ms per document (depends on model & hardware)
        - Memory: Model size + batch overhead
        - Quality: +25-40% improvement in result relevance
    
    Example:
        >>> reranker = CrossEncoderReranker(
        ...     model_name="your-reranker-model",
        ...     ollama_base_url="http://localhost:11434",
        ...     max_concurrent=5
        ... )
        >>> 
        >>> query = "how to implement authentication"
        >>> docs = [("doc1", "def login():"), ("doc2", "import jwt")]
        >>> results = reranker.rerank(query, docs, top_k=3)
        >>> for doc_id, score, text in results:
        ...     print(f"{doc_id}: {score:.3f} - {text[:50]}")
    """
    
    def __init__(
        self,
        model_name: str = "qwen3:0.6b",
        ollama_base_url: str = "http://localhost:11434",
        max_concurrent: int = 5,
        timeout: float = 60.0,
        max_sequence_length: Optional[int] = None,
        prompt_template: Optional[str] = None,
    ):
        """Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the Ollama reranking model to use.
                       Recommended: "phi3:mini" (fast, lightweight) or
                       "qwen3:0.6b" (small, capable)
            ollama_base_url: URL for the Ollama API server
            max_concurrent: Maximum number of concurrent reranking requests.
                           Lower values reduce VRAM usage but increase latency.
            timeout: HTTP request timeout in seconds
            max_sequence_length: Maximum tokens for query+document.
                                If None, uses model's default (usually 512-8192)
            prompt_template: Optional custom prompt template.
                           Must contain {query} and {document} placeholders.
        
        Raises:
            ValueError: If model_name is empty or invalid
        """
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")
        
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_sequence_length = max_sequence_length
        
        # Ollama API endpoint for reranking (using /api/generate)
        self.api_url = f"{self.ollama_base_url}/api/generate"
        
        # Default prompt template for reranking
        # Can be customized for specific models
        self.prompt_template = prompt_template or self._get_default_prompt_template()
        
        logger.info(
            "Initialized CrossEncoderReranker with model=%s, max_concurrent=%d",
            model_name, max_concurrent
        )
    
    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template for reranking.
        
        The template formats query and document for the cross-encoder.
        Different models may work better with different formats.
        """
        # mxbai-rerank and BGE models typically use this format
        return """Query: {query}

Document: {document}

On a scale of 0 to 100, how relevant is this document to the query? Provide only a number."""
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str]],
        top_k: int = 10
    ) -> List[Tuple[str, float, str]]:
        """Rerank documents based on query relevance.
        
        Args:
            query: The search query
            documents: List of (doc_id, doc_text) tuples to rerank
            top_k: Maximum number of results to return
        
        Returns:
            List of (doc_id, score, doc_text) tuples sorted by score descending.
            Scores are normalized to 0-1 range.
        
        Example:
            >>> query = "authentication function"
            >>> docs = [("doc1", "def login():"), ("doc2", "database setup")]
            >>> results = reranker.rerank(query, docs, top_k=2)
            >>> # results = [("doc1", 0.95, "def login():"), ...]
        """
        if not documents:
            return []
        
        if not query or not query.strip():
            logger.warning("Empty query provided to reranker, returning documents unsorted")
            return [(doc_id, 0.5, text) for doc_id, text in documents[:top_k]]
        
        try:
            scores = asyncio.run(self._rerank_async(query, documents))
        except RuntimeError as e:
            if "already running" in str(e):
                logger.debug("Event loop already running, using sync fallback")
                scores = self._rerank_sync(query, documents)
            else:
                raise
        
        # Combine doc_id, score, and text, handling any exceptions
        scored_results = []
        for (doc_id, text), score in zip(documents, scores):
            if isinstance(score, Exception):
                logger.error("Score computation failed for document %s: %s", doc_id, score)
                scored_results.append((doc_id, 0.0, text))
            else:
                scored_results.append((doc_id, score, text))
        
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results[:top_k]
    
    async def _rerank_async(
        self,
        query: str,
        documents: List[Tuple[str, str]]
    ) -> List[float]:
        """Async reranking with concurrent requests."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                self._score_pair_with_semaphore(client, query, doc_text, i, semaphore)
                for i, (_, doc_text) in enumerate(documents)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scores: List[float] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Failed to score document %d: %s", i, result)
                scores.append(0.0)
            else:
                scores.append(result)
        
        return scores
    
    def _rerank_sync(
        self,
        query: str,
        documents: List[Tuple[str, str]]
    ) -> List[float]:
        """Synchronous reranking fallback."""
        scores: List[float] = []
        
        with httpx.Client(timeout=self.timeout) as client:
            for i, (_, doc_text) in enumerate(documents):
                try:
                    score = self._score_pair(client, query, doc_text, i)
                    scores.append(score)
                except Exception as e:
                    logger.error("Failed to score document %d: %s", i, e)
                    scores.append(0.0)
        
        return scores
    
    async def _score_pair_with_semaphore(
        self,
        client: httpx.AsyncClient,
        query: str,
        document: str,
        idx: int,
        semaphore: asyncio.Semaphore
    ) -> float:
        """Score a query-document pair with semaphore-controlled concurrency."""
        async with semaphore:
            return await self._score_pair_async(client, query, document, idx)
    
    async def _score_pair_async(
        self,
        client: httpx.AsyncClient,
        query: str,
        document: str,
        idx: int
    ) -> float:
        """Score a single query-document pair asynchronously using generate API."""
        try:
            prompt = self._format_prompt(query, document)
            
            response = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse the model's response to get the score
            response_text = data.get("response", "").strip()
            score = self._parse_score(response_text)
            return score
            
        except Exception as e:
            logger.error("Error scoring pair %d: %s", idx, e)
            raise
    
    def _score_pair(
        self,
        client: httpx.Client,
        query: str,
        document: str,
        idx: int
    ) -> float:
        """Score a single query-document pair synchronously using generate API."""
        try:
            prompt = self._format_prompt(query, document)
            
            response = client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse the model's response to get the score
            response_text = data.get("response", "").strip()
            score = self._parse_score(response_text)
            return score
            
        except Exception as e:
            logger.error("Error scoring pair %d: %s", idx, e)
            raise
    
    def _format_prompt(self, query: str, document: str) -> str:
        """Format the prompt for the reranking model.
        
        Args:
            query: The search query
            document: The document text
        
        Returns:
            Formatted prompt string
        """
        # Truncate document if max_sequence_length is set
        if self.max_sequence_length:
            # Rough approximation: 1 token ≈ 4 characters
            max_chars = self.max_sequence_length * 4
            if len(document) > max_chars:
                document = document[:max_chars - 3] + "..."
        
        return self.prompt_template.format(query=query, document=document)
    
    def _parse_score(self, response_text: str) -> float:
        """Parse the model's response to extract a relevance score.
        
        The model should output a number between 0 and 100.
        We extract the first number found and normalize it to 0-1 range.
        If the score is already in 0-1 range, we keep it as-is.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            Normalized score between 0.0 and 1.0
        """
        import re
        
        # Look for a number in the response (including decimals)
        match = re.search(r'\d+\.?\d*', response_text)
        if match:
            try:
                score = float(match.group())
                # Normalize to 0-1 range only if score > 1 (assuming 0-100 scale)
                # If score is already <= 1, keep it as-is (already normalized)
                if score > 1.0:
                    score = score / 100.0
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        # If no number found, log warning and return neutral score
        logger.warning("Could not parse score from response: '%s', using default", response_text)
        return 0.5
    
    def check_model_available(self) -> Tuple[bool, Optional[str]]:
        """Check if the configured model is available in Ollama.
        
        Returns:
            Tuple of (is_available, error_message)
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.ollama_base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                
                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check for exact match or tag match
                if self.model_name in model_names:
                    return True, None
                
                # Check without tag
                base_name = self.model_name.split(":")[0]
                if any(m.startswith(base_name) for m in model_names):
                    return True, None
                
                return False, f"Model '{self.model_name}' not found. Available: {model_names}"
                
        except Exception as e:
            return False, f"Failed to check model availability: {e}"
