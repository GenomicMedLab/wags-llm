"""Registry.

Maps (name, version) -> template instance.

Users typically:
* create prompts or skills in their project
* register them here or pass a custom registry
"""

import logging

from wags_llm.templates.base import PromptTemplate
from wags_llm.templates.skill_template import SkillTemplate

_logger = logging.getLogger(__name__)


class Registry:
    """Store and retrieve prompt and skill templates."""

    def __init__(self) -> None:
        """Initialize an empty template registry."""
        self._templates: dict[tuple[str, str], PromptTemplate | SkillTemplate] = {}

    def register(self, template: PromptTemplate | SkillTemplate) -> None:
        """Register a template.

        :param template: Template instance to register.
        """
        _logger.debug(
            "Registering template: name='%s', version='%s'",
            template.name,
            template.version,
        )
        self._templates[(template.name, template.version)] = template

    def get(self, name: str, version: str) -> PromptTemplate | SkillTemplate:
        """Retrieve a template by name and version.

        :param name: Template name.
        :param version: Template version.
        :return: Registered template.
        :raise KeyError: If template not found.
        """
        try:
            return self._templates[(name, version)]
        except KeyError as exc:
            msg = f"Template not found: ({name}, {version})"
            _logger.exception(msg)
            raise KeyError(msg) from exc


def build_empty_registry() -> Registry:
    """Create an empty registry.

    :return: New Registry instance.
    """
    return Registry()
