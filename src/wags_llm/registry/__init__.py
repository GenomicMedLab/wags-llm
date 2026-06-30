"""Registry module.

Store and retrieve versioned prompt and skill templates.
"""

from wags_llm.registry.base import Registry, build_empty_registry

__all__ = [
    "Registry",
    "build_empty_registry",
]
