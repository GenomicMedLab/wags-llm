"""Test that InMemoryCache works correctly"""

from wags_llm.cache.in_memory import InMemoryCache


def test_in_memory_cache_get_set():
    """Test that get and set methods work correctly"""
    cache = InMemoryCache()
    value = {"a": 1}

    cache.set("key1", value)

    assert cache.get("key1") == value


def test_in_memory_cache_missing_key():
    """Test that get works correctly when key is not in the cache"""
    cache = InMemoryCache()

    assert cache.get("missing") is None


def test_delete_removes_existing_key():
    """Test that delete method works correctly"""
    cache = InMemoryCache()

    cache.set("a", {"value": 1})
    assert len(cache) == 1

    cache.delete("a")
    assert len(cache) == 0
    assert cache.get("a") is None


def test_delete_missing_key_no_error():
    """Test that delete doesn't raise an error if a key doesn't exist"""
    cache = InMemoryCache()

    cache.set("a", {"value": 1})
    cache.delete("missing")

    assert len(cache) == 1
    assert cache.get("a") == {"value": 1}


def test_clear_removes_all_entries():
    """Test that remove all works correctly"""
    cache = InMemoryCache()

    cache.set("a", {"value": 1})
    cache.set("b", {"value": 2})

    assert len(cache) == 2

    cache.clear()

    assert len(cache) == 0
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_len_reflects_cache_size():
    """Test that len works correctly"""
    cache = InMemoryCache()

    assert len(cache) == 0

    cache.set("a", {"value": 1})
    assert len(cache) == 1

    cache.set("b", {"value": 2})
    assert len(cache) == 2

    cache.delete("a")
    assert len(cache) == 1

    cache.clear()
    assert len(cache) == 0


def test_len_respects_max_entries_eviction():
    """Test max entries is being followed"""
    cache = InMemoryCache(max_entries=2)

    cache.set("a", {"value": 1})
    cache.set("b", {"value": 2})
    assert cache.get("a") is not None
    assert cache.get("b") is not None
    assert cache.get("c") is None

    cache.set("c", {"value": 3})  # should remove one

    assert len(cache) == 2
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None
