"""Prompt interfaces and registry.

Define and manage versioned prompt and skill templates.
"""

from wags_llm.templates.base import BaseTemplate, TemplateType
from wags_llm.templates.prompt_template import PromptTemplate
from wags_llm.templates.skill_template import SkillTemplate

__all__ = [
    "BaseTemplate",
    "PromptTemplate",
    "SkillTemplate",
    "TemplateType",
]
