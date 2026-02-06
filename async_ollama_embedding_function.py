"""Async Ollama embedding function for ChromaDB.

This module provides an async-enabled embedding function with concurrent
batch processing for faster indexing with Ollama embeddings.
"""

import asyncio
import logging
from typing import List

import httpx
from chromadb.utils.embedding_functions import EmbeddingFunction

logger = logging.getLogger(__name__)


class AsyncOllamaEmbeddingFunction(EmbeddingFunction):
    """Async-enabled Ollama embedding function with concurrent batch processing.
    
    This embedding function significantly speeds up indexing by processing
    multiple embedding requests concurrently using asyncio and httpx.
    
    Features:
        - Concurrent embedding requests (configurable concurrency limit)
        - Batch processing to reduce HTTP overhead
        - Same quality embeddings as the standard OllamaEmbeddingFunction
        - Automatic fallback to sync processing if needed
    
    Example:
        >>> ef = AsyncOllamaEmbeddingFunction(
        ...     model_name="nomic-embed-text",
        ...     url="http://localhost:11434",
        ...     max_concurrent=10,
        ...     batch_size=32
        ... )
        >>> embeddings = ef(["text1", "text2", "text3"])
    """
    
    def __init__(
        self,
        model_name: str,
        url: str,
        max_concurrent: int,
        batch_size: int,
        timeout: float = 120.0
    ):
        """Initialize the async Ollama embedding function.
        
        Args:
            model_name: Name of the Ollama embedding model
            url: Base URL for Ollama API
            max_concurrent: Maximum number of concurrent embedding requests
            batch_size: Number of texts to embed in a single API call
            timeout: HTTP request timeout in seconds
        """
        self.model_name = model_name
        self.url = url.rstrip("/")
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.timeout = timeout
        self.api_url = f"{self.url}/api/embed"
        
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            return []
        
        try:
            return asyncio.run(self._embed_async(texts))
        except RuntimeError as e:
            if "already running" in str(e):
                logger.debug("Event loop already running, using sync batch processing")
                return self._embed_sync_fallback(texts)
            raise
    
    async def _embed_async(self, texts: List[str]) -> List[List[float]]:
        """Async embedding with concurrent batch processing."""
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                self._embed_batch_with_semaphore(client, batch, i, semaphore)
                for i, batch in enumerate(batches)
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_embeddings: List[List[float]] = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error("Batch %d failed: %s", i, result)
                all_embeddings.extend([[0.0] * 768] * len(batches[i]))
            else:
                all_embeddings.extend(result)
        
        return all_embeddings
    
    async def _embed_batch_with_semaphore(
        self,
        client: httpx.AsyncClient,
        batch: List[str],
        batch_idx: int,
        semaphore: asyncio.Semaphore
    ) -> List[List[float]]:
        """Embed a batch with semaphore-controlled concurrency."""
        async with semaphore:
            return await self._embed_batch(client, batch, batch_idx)
    
    async def _embed_batch(
        self,
        client: httpx.AsyncClient,
        batch: List[str],
        batch_idx: int
    ) -> List[List[float]]:
        """Embed a single batch of texts."""
        try:
            response = await client.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "input": batch
                }
            )
            response.raise_for_status()
            data = response.json()
            
            if "embeddings" in data:
                embeddings = data["embeddings"]
                if embeddings and isinstance(embeddings[0], list):
                    return embeddings
                return [embeddings]
            else:
                return [data.get("embedding", [])]
                
        except Exception as e:
            logger.error("Error embedding batch %d: %s", batch_idx, e)
            raise
    
    def _embed_sync_fallback(self, texts: List[str]) -> List[List[float]]:
        """Synchronous fallback using httpx for batch processing."""
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        
        all_embeddings: List[List[float]] = []
        
        with httpx.Client(timeout=self.timeout) as client:
            for i, batch in enumerate(batches):
                try:
                    response = client.post(
                        self.api_url,
                        json={
                            "model": self.model_name,
                            "input": batch
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if "embeddings" in data:
                        embeddings = data["embeddings"]
                        if embeddings and isinstance(embeddings[0], list):
                            all_embeddings.extend(embeddings)
                        else:
                            all_embeddings.append(embeddings)
                    else:
                        all_embeddings.append(data.get("embedding", []))
                        
                except Exception as e:
                    logger.error("Error embedding batch %d: %s", i, e)
                    all_embeddings.extend([[0.0] * 768] * len(batch))
        
        return all_embeddings
