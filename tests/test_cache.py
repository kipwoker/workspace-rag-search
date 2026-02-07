"""Tests for query caching functionality."""

import json
import time
from unittest.mock import Mock, patch

import pytest

from cache import QueryCache, CacheEntry, CacheStats


class TestQueryCache:
    """Test cases for QueryCache class."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        cache = QueryCache(max_size=50, ttl_seconds=300, enabled=True)
        assert cache.max_size == 50
        assert cache.ttl_seconds == 300
        assert cache.enabled is True
        assert len(cache) == 0

    def test_cache_disabled(self):
        """Test cache operations when disabled."""
        cache = QueryCache(enabled=False)
        
        # Should not store
        cache.set("query", 5, None, 60, True, '{"result": "test"}')
        assert len(cache) == 0
        
        # Should return None
        result = cache.get("query", 5, None, 60, True)
        assert result is None

    def test_cache_store_and_retrieve(self):
        """Test storing and retrieving from cache."""
        cache = QueryCache(max_size=10)
        
        # Store result
        cache.set("test query", 5, None, 60, True, '{"count": 3}')
        assert len(cache) == 1
        
        # Retrieve result
        result = cache.get("test query", 5, None, 60, True)
        assert result == '{"count": 3}'

    def test_cache_case_insensitive(self):
        """Test cache keys are case-insensitive."""
        cache = QueryCache(max_size=10)
        
        cache.set("Test Query", 5, None, 60, True, '{"result": "found"}')
        
        # Should find with different case
        result = cache.get("test query", 5, None, 60, True)
        assert result == '{"result": "found"}'

    def test_cache_different_params(self):
        """Test different query parameters create different cache entries."""
        cache = QueryCache(max_size=10)
        
        cache.set("query", 5, None, 60, True, '{"limit": 5}')
        cache.set("query", 10, None, 60, True, '{"limit": 10}')
        
        assert len(cache) == 2
        assert cache.get("query", 5, None, 60, True) == '{"limit": 5}'
        assert cache.get("query", 10, None, 60, True) == '{"limit": 10}'

    def test_cache_ttl_expiration(self):
        """Test cache entries expire after TTL."""
        cache = QueryCache(max_size=10, ttl_seconds=0.1)
        
        cache.set("query", 5, None, 60, True, '{"data": "test"}')
        
        # Should be available immediately
        assert cache.get("query", 5, None, 60, True) is not None
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("query", 5, None, 60, True) is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = QueryCache(max_size=3)
        
        # Fill cache
        cache.set("query1", 5, None, 60, True, '{"n": 1}')
        cache.set("query2", 5, None, 60, True, '{"n": 2}')
        cache.set("query3", 5, None, 60, True, '{"n": 3}')
        
        assert len(cache) == 3
        
        # Access query1 to make it recently used
        cache.get("query1", 5, None, 60, True)
        
        # Add new entry, should evict query2 (least recently used)
        cache.set("query4", 5, None, 60, True, '{"n": 4}')
        
        assert len(cache) == 3
        assert cache.get("query1", 5, None, 60, True) is not None  # Still there
        assert cache.get("query2", 5, None, 60, True) is None  # Evicted
        assert cache.get("query3", 5, None, 60, True) is not None
        assert cache.get("query4", 5, None, 60, True) is not None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = QueryCache(max_size=10)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        
        # Store and retrieve
        cache.set("query", 5, None, 60, True, '{"test": true}')
        cache.get("query", 5, None, 60, True)  # hit
        cache.get("other", 5, None, 60, True)  # miss
        
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    def test_cache_invalidate_all(self):
        """Test clearing all cache entries."""
        cache = QueryCache(max_size=10)
        
        cache.set("query1", 5, None, 60, True, '{"n": 1}')
        cache.set("query2", 5, None, 60, True, '{"n": 2}')
        
        cleared = cache.invalidate()
        
        assert cleared == 2
        assert len(cache) == 0

    def test_cache_invalidate_pattern(self):
        """Test clearing cache entries matching pattern."""
        cache = QueryCache(max_size=10)
        
        cache.set("auth middleware", 5, None, 60, True, '{"n": 1}')
        cache.set("auth login", 5, None, 60, True, '{"n": 2}')
        cache.set("user profile", 5, None, 60, True, '{"n": 3}')
        
        cleared = cache.invalidate("auth")
        
        assert cleared == 2
        assert len(cache) == 1
        assert cache.get("user profile", 5, None, 60, True) is not None

    def test_cache_update_existing(self):
        """Test updating an existing cache entry."""
        cache = QueryCache(max_size=10)
        
        cache.set("query", 5, None, 60, True, '{"v": 1}')
        cache.set("query", 5, None, 60, True, '{"v": 2}')
        
        assert len(cache) == 1
        result = json.loads(cache.get("query", 5, None, 60, True))
        assert result["v"] == 2

    def test_cache_entry_touch(self):
        """Test cache entry touch updates stats."""
        entry = CacheEntry(query="test", result='{"r": 1}')
        
        assert entry.access_count == 0
        assert entry.cache_hits == 0
        
        entry.touch()
        
        assert entry.access_count == 1
        assert entry.cache_hits == 1
        assert entry.last_accessed >= entry.timestamp

    def test_cache_stats_to_dict(self):
        """Test CacheStats to_dict method."""
        stats = CacheStats(
            hits=10,
            misses=5,
            evictions=2,
            size=8,
            max_size=10,
            hit_rate=0.6667,
            total_queries=15,
            avg_entry_age_ms=1000.5
        )
        
        d = stats.to_dict()
        
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert d["size"] == 8
        assert d["max_size"] == 10
        assert d["hit_rate"] == 0.6667
        assert d["total_queries"] == 15
        assert d["avg_entry_age_ms"] == 1000.5

    def test_cache_thread_safety(self):
        """Test cache operations are thread-safe."""
        import threading
        
        cache = QueryCache(max_size=100)
        errors = []
        
        def writer():
            try:
                for i in range(100):
                    cache.set(f"query{i}", 5, None, 60, True, f'{{"n": {i}}}')
            except Exception as e:
                errors.append(f"write: {e}")
        
        def reader():
            try:
                for i in range(100):
                    cache.get(f"query{i}", 5, None, 60, True)
            except Exception as e:
                errors.append(f"read: {e}")
        
        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"


class TestCacheIntegration:
    """Integration tests for cache with search tool."""

    def test_cache_cleared_on_refresh(self, tmp_path):
        """Test that cache is cleared when index is refreshed."""
        from workspace_rag_search_tool import WorkspaceRagSearchTool
        from unittest.mock import Mock, patch
        
        mock_collection = Mock()
        
        with patch.object(WorkspaceRagSearchTool, '_get_or_create_collection', return_value=mock_collection):
            with patch.object(WorkspaceRagSearchTool, '_initialize_reranker'):
                with patch.object(WorkspaceRagSearchTool, '_index_files'):
                    tool = WorkspaceRagSearchTool(str(tmp_path))
                    tool._collection = mock_collection
                    tool._gitignore_parser = Mock()
                    
                    # Manually set up cache
                    tool._cache = QueryCache(max_size=10, enabled=True)
                    
                    # Add something to cache
                    tool._cache.set("query", 5, None, 60, True, '{"test": true}')
                    assert len(tool._cache) == 1
                    
                    # Refresh should clear cache
                    result = json.loads(tool.refresh_index())
                    assert result["cache_cleared"] == 1
                    assert len(tool._cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
