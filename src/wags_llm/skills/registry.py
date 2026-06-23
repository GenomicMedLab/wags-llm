"""skill registry.

Maps (name, version) -> skill instance.

Users typically:
* create skills in their project
* register them here or pass a custom registry
"""

import logging

from wags_llm.skills.base import BaseSkillTemplate

_logger = logging.getLogger(__name__)


class SkillRegistry:
    """Store and retrieve skill."""

    def __init__(self) -> None:
        """Initialize an empty skill registry."""
        self._skills: dict[tuple[str, str], BaseSkillTemplate] = {}

    def register(self, skill: BaseSkillTemplate) -> None:
        """Register a skill.

        :param skill: skill instance to register.
        """
        _logger.debug(
            "Registering skill: name='%s', version='%s'", skill.name, skill.version
        )
        self._skills[(skill.name, skill.version)] = skill

    def get(self, name: str, version: str) -> BaseSkillTemplate:
        """Retrieve a skill by name and version.

        :param name: Skill name.
        :param version: Skill version.
        :return: Registered skill.
        :raise KeyError: If skill is not found.
        """
        try:
            return self._skills[(name, version)]
        except KeyError as exc:
            msg = f"Skill not found: ({name}, {version})"
            _logger.exception(msg)
            raise KeyError(msg) from exc


def build_empty_registry() -> SkillRegistry:
    """Create an empty skill registry.

    :return: New SkillRegistry instance.
    """
    return SkillRegistry()
