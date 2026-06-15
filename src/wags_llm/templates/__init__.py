"""Prompt interfaces and registry.

Define and manage versioned prompt templates.
"""

from wags_llm.templates.base import PromptTemplate
from wags_llm.templates.skill_template import SkillTemplate

__all__ = [
    "PromptTemplate",
    "SkillTemplate",
]
