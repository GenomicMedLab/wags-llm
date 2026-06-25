"""Registry module.

Store and retrieve versioned prompt and skill templates.
"""

from wags_llm.registry.base import Registry, TaskType, build_empty_registry

__all__ = [
    "Registry",
    "TaskType",
    "build_empty_registry",
]
