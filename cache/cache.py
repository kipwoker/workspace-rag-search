"""LRU Cache implementation for query results.

This module provides an in-memory LRU cache for search query results,
helping to speed up repeated searches significantly.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry containing query result and metadata.
    
    Attributes:
        query: The original search query
        result: Cached result (JSON string)
        timestamp: When the entry was created
        access_count: Number of times this entry was accessed
        last_accessed: Timestamp of last access
        cache_hits: Number of cache hits for this entry
    """
    query: str
    result: str
    timestamp: float = field(default_factory=time.time)
    access_count: int = field(default=0)
    last_accessed: float = field(default_factory=time.time)
    cache_hits: int = field(default=0)
    
    def touch(self) -> None:
        """Update access statistics when entry is accessed."""
        self.access_count += 1
        self.cache_hits += 1
        self.last_accessed = time.time()


@dataclass
class CacheStats:
    """Cache statistics for monitoring and debugging.
    
    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of entries evicted due to size limit
        size: Current number of entries in cache
        max_size: Maximum cache size
        hit_rate: Cache hit rate (0.0 to 1.0)
        total_queries: Total number of queries processed
        avg_entry_age_ms: Average age of cache entries in milliseconds
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    total_queries: int = 0
    avg_entry_age_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for JSON serialization."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": round(self.hit_rate, 4),
            "total_queries": self.total_queries,
            "avg_entry_age_ms": round(self.avg_entry_age_ms, 2)
        }


class QueryCache:
    """Thread-safe LRU cache for search query results.
    
    This cache stores search results keyed by query parameters
    (query string, limit, path_filter, etc.) to speed up repeated searches.
    
    The cache uses an OrderedDict to maintain LRU order and provides
    thread-safe operations using a lock.
    
    Example:
        >>> cache = QueryCache(max_size=100, ttl_seconds=300)
        >>> cache.set("auth middleware", 5, None, '{"results": [...]}')
        >>> result = cache.get("auth middleware", 5, None)
        >>> print(cache.get_stats())
    
    Attributes:
        max_size: Maximum number of entries to store
        ttl_seconds: Time-to-live in seconds (None for no expiration)
        enabled: Whether caching is enabled
    """
    
    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: Optional[int] = None,
        enabled: bool = True
    ):
        """Initialize the query cache.
        
        Args:
            max_size: Maximum number of cache entries (default: 100)
            ttl_seconds: Entry TTL in seconds, None for no expiration (default: None)
            enabled: Whether caching is enabled (default: True)
        """
        self.max_size = max(max_size, 1)  # Ensure at least 1
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        
        # OrderedDict maintains insertion order - used for LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.debug(
            "QueryCache initialized (max_size=%d, ttl=%s, enabled=%s)",
            self.max_size,
            self.ttl_seconds if self.ttl_seconds else "none",
            self.enabled
        )
    
    def _make_key(
        self,
        query: str,
        limit: int,
        path_filter: Optional[str],
        rrf_k: int,
        rerank_enabled: bool,
        hyde_enabled: bool = False
    ) -> str:
        """Create a unique cache key from query parameters.
        
        The key includes all parameters that affect search results
        to ensure cache correctness.
        
        Args:
            query: Search query string
            limit: Result limit
            path_filter: Optional path filter
            rrf_k: RRF constant
            rerank_enabled: Whether reranking is enabled
            hyde_enabled: Whether HyDE query expansion is enabled
            
        Returns:
            String key for cache lookup
        """
        # Normalize query for consistent keys
        normalized_query = query.strip().lower()
        path_filter_str = path_filter.lower() if path_filter else ""
        
        # Create deterministic key
        key_parts = [
            f"q:{normalized_query}",
            f"l:{limit}",
            f"p:{path_filter_str}",
            f"rrf:{rrf_k}",
            f"rnk:{rerank_enabled}",
            f"hyde:{hyde_enabled}"
        ]
        return "|".join(key_parts)
    
    def get(
        self,
        query: str,
        limit: int,
        path_filter: Optional[str],
        rrf_k: int,
        rerank_enabled: bool,
        hyde_enabled: bool = False
    ) -> Optional[str]:
        """Get cached result if available and not expired.
        
        Args:
            query: Search query string
            limit: Result limit used
            path_filter: Optional path filter used
            rrf_k: RRF constant used
            rerank_enabled: Whether reranking was enabled
            hyde_enabled: Whether HyDE query expansion was enabled
            
        Returns:
            Cached result string or None if not found/expired
        """
        if not self.enabled:
            return None
        
        key = self._make_key(query, limit, path_filter, rrf_k, rerank_enabled, hyde_enabled)
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                logger.debug("Cache miss for key: %s", key[:50])
                return None
            
            # Check TTL expiration
            if self.ttl_seconds is not None:
                age = time.time() - entry.timestamp
                if age > self.ttl_seconds:
                    # Entry expired - remove it
                    del self._cache[key]
                    self._misses += 1
                    logger.debug("Cache entry expired for key: %s", key[:50])
                    return None
            
            # Cache hit - update LRU order and stats
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            
            logger.debug(
                "Cache hit for key: %s (hits: %d)",
                key[:50],
                entry.cache_hits
            )
            
            return entry.result
    
    def set(
        self,
        query: str,
        limit: int,
        path_filter: Optional[str],
        rrf_k: int,
        rerank_enabled: bool,
        result: str,
        hyde_enabled: bool = False
    ) -> None:
        """Store result in cache.
        
        Args:
            query: Search query string
            limit: Result limit used
            path_filter: Optional path filter used
            rrf_k: RRF constant used
            rerank_enabled: Whether reranking was enabled
            hyde_enabled: Whether HyDE query expansion was enabled
            result: Result string to cache
        """
        if not self.enabled:
            return
        
        key = self._make_key(query, limit, path_filter, rrf_k, rerank_enabled, hyde_enabled)
        
        with self._lock:
            # If key exists, update it and move to end
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key].result = result
                self._cache[key].timestamp = time.time()
                logger.debug("Cache entry updated for key: %s", key[:50])
            else:
                # Check if we need to evict
                if len(self._cache) >= self.max_size:
                    # Remove oldest (first) item
                    oldest_key, oldest_entry = self._cache.popitem(last=False)
                    self._evictions += 1
                    logger.debug(
                        "Cache eviction: removed key %s (had %d hits)",
                        oldest_key[:50],
                        oldest_entry.cache_hits
                    )
                
                # Add new entry
                entry = CacheEntry(query=query, result=result)
                self._cache[key] = entry
                logger.debug("Cache entry added for key: %s", key[:50])
    
    def invalidate(self, query_pattern: Optional[str] = None) -> int:
        """Invalidate cache entries.
        
        Args:
            query_pattern: If provided, only invalidate entries matching this pattern
                          (substring match, case-insensitive). If None, clear all.
                          
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if query_pattern is None:
                count = len(self._cache)
                self._cache.clear()
                logger.info("Cache cleared: %d entries removed", count)
                return count
            
            # Pattern-based invalidation
            pattern_lower = query_pattern.lower()
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if pattern_lower in entry.query.lower()
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
            
            logger.info(
                "Cache invalidation: %d entries matching '%s' removed",
                len(keys_to_remove),
                query_pattern
            )
            return len(keys_to_remove)
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics.
        
        Returns:
            CacheStats object with current statistics
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            # Calculate average entry age
            if self._cache:
                now = time.time()
                total_age = sum(now - entry.timestamp for entry in self._cache.values())
                avg_age_ms = (total_age / len(self._cache)) * 1000
            else:
                avg_age_ms = 0.0
            
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._cache),
                max_size=self.max_size,
                hit_rate=hit_rate,
                total_queries=total,
                avg_entry_age_ms=avg_age_ms
            )
    
    def get_cache_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about cached entries.
        
        Returns:
            List of dictionaries with entry information
        """
        with self._lock:
            info = []
            now = time.time()
            
            for entry in self._cache.values():
                age_ms = (now - entry.timestamp) * 1000
                info.append({
                    "query": entry.query[:50] + "..." if len(entry.query) > 50 else entry.query,
                    "access_count": entry.access_count,
                    "cache_hits": entry.cache_hits,
                    "age_ms": round(age_ms, 2),
                    "result_size": len(entry.result)
                })
            
            # Sort by last accessed (most recent first)
            info.sort(key=lambda x: -x["cache_hits"])
            return info
    
    def clear_stats(self) -> None:
        """Reset cache statistics (preserves entries)."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            logger.debug("Cache statistics reset")
    
    def __len__(self) -> int:
        """Return current number of cache entries."""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, key_tuple: Tuple[str, int, Optional[str], int, bool, bool]) -> bool:
        """Check if a query is in cache.
        
        Args:
            key_tuple: (query, limit, path_filter, rrf_k, rerank_enabled, hyde_enabled)
            
        Returns:
            True if in cache and not expired
        """
        if not self.enabled or len(key_tuple) != 6:
            return False
        
        query, limit, path_filter, rrf_k, rerank_enabled, hyde_enabled = key_tuple
        key = self._make_key(query, limit, path_filter, rrf_k, rerank_enabled, hyde_enabled)
        
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            
            # Check expiration
            if self.ttl_seconds is not None:
                age = time.time() - entry.timestamp
                if age > self.ttl_seconds:
                    return False
            
            return True
