"""Cache interface for storing LLM task results."""

from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):
    """Abstract base class for LLM result caches.

    Cache implementations are expected to store JSON-serializable values keyed by unique
    strings.
    """

    @abstractmethod
    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve a cached value by key.

        :param key: Cache key.
        :returns: Cached value if present, otherwise `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store a value in the cache.

        :param key: Cache key.
        :param value: JSON-serializable value to cache.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a value from the cache if present.

        :param key: Cache key.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all cached entries."""
        raise NotImplementedError
