"""Prompt interfaces and registry.

Define and manage versioned prompt templates.
"""

from wags_llm.registry.base import Registry, build_empty_registry

__all__ = [
    "Registry",
    "build_empty_registry",
]
