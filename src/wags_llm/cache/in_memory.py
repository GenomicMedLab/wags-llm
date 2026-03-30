"""In-memory LRU cache implementation.

Useful for local runs and testing.
"""

import logging
from collections import OrderedDict
from typing import Any

from wags_llm.cache.base import BaseCache

_logger = logging.getLogger(__name__)


class InMemoryCache(BaseCache):
    """In-memory LRU cache implementation."""

    def __init__(self, max_entries: int | None = None) -> None:
        """Initialize the in-memory cache.

        :param max_entries: Maximum number of entries to retain. If `None`, the cache is
            unbounded.
        """
        self.max_entries = max_entries
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve a cached value by key.

        Accessing a key that already exists will mark it as recently used.

        :param key: Cache key.
        :returns: Cached value if present, otherwise `None`.
        """
        value = self._store.get(key)
        if value is None:
            _logger.debug("Cache miss for cache key='%s'", key)
            return None

        self._store.move_to_end(key)
        _logger.debug("Cache hit for cache key='%s'", key)
        return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store a value in the cache.

        If `max_entries` is configured and the cache grows beyond that
        size, the least recently used entry is evicted.

        :param key: Cache key.
        :param value: JSON-serializable value to cache.
        """
        self._store[key] = value
        self._store.move_to_end(key)

        if self.max_entries is not None:
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        """Remove a value from the cache if present.

        :param key: Cache key.
        """
        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        """Return the number of cached entries.

        :returns: Number of items in the cache.
        """
        return len(self._store)
