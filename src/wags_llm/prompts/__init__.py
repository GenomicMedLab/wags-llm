"""Prompt interfaces and registry.

Define and manage versioned prompt templates.
"""

from wags_llm.prompts.base import BasePromptTemplate
from wags_llm.prompts.registry import PromptRegistry, build_empty_registry

__all__ = [
    "BasePromptTemplate",
    "PromptRegistry",
    "build_empty_registry",
]
