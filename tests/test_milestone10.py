import pytest
from adaptive_rag.storage.cache import MultiTierCache

def test_cache_legacy_compatibility():
    cache = MultiTierCache()
    cache.set_query_response("query", "doc1", {"answer": "test"})
    assert cache.get_query_response("query", "doc1") == {"answer": "test"}
