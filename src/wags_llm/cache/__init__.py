"""Cache implementations for LLM tasks."""

from wags_llm.cache.base import BaseCache
from wags_llm.cache.in_memory import InMemoryCache

__all__ = ["BaseCache", "InMemoryCache"]
