"""Skill interfaces and registry.

Define and manage versioned skill templates.
"""

from wags_llm.skills.base import BaseSkillTemplate
from wags_llm.skills.registry import SkillRegistry, build_empty_registry

__all__ = [
    "BaseSkillTemplate",
    "SkillRegistry",
    "build_empty_registry",
]
