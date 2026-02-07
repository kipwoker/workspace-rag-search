"""Query result caching for workspace RAG search.

Provides LRU (Least Recently Used) cache for search query results
to speed up repeated searches.
"""

from cache.cache import QueryCache, CacheEntry, CacheStats

__all__ = ["QueryCache", "CacheEntry", "CacheStats"]
