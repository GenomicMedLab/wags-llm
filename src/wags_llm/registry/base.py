"""Registry.

Maps (name, version, task_type) -> template instance.
Template instances can be either prompts or skills.

Users typically:
* create prompts or skills in their project
* register them here or pass a custom registry
"""

import logging
from enum import Enum

from wags_llm.templates.base import PromptTemplate
from wags_llm.templates.skill_template import SkillTemplate

_logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enum for task types supported by StructuredTaskRunner."""

    SKILL = "skill"
    PROMPT = "prompt"


class Registry:
    """Store and retrieve prompt and skill templates."""

    def __init__(self) -> None:
        """Initialize an empty template registry."""
        self._templates: dict[
            tuple[str, str, TaskType], PromptTemplate | SkillTemplate
        ] = {}

    def register(self, template: PromptTemplate | SkillTemplate) -> None:
        """Register a template.

        :param template: Template instance to register.
        """
        task_type = self._get_task_type(template)

        key = (template.name, template.version, task_type)

        _logger.debug(
            "Registering template: name='%s', version='%s', task_type='%s'",
            template.name,
            template.version,
            task_type.value,
        )

        if key in self._templates:
            msg = f"Template already registered:({template.name}, {template.version}, {task_type.value})"
            _logger.error(msg)
            raise ValueError(msg)

        self._templates[key] = template

    def get(
        self,
        name: str,
        version: str,
        task_type: TaskType,
    ) -> PromptTemplate | SkillTemplate:
        """Retrieve a template by name and version.

        :param name: Template name.
        :param version: Template version.
        :param task_type: Template type.
        :return: Registered template.
        :raise KeyError: If template not found.
        """
        key = (name, version, task_type)

        try:
            return self._templates[key]
        except KeyError as exc:
            msg = f"Template not found: ({name}, {version}, {task_type.value})"
            _logger.exception(msg)
            raise KeyError(msg) from exc

    def _get_task_type(self, template: PromptTemplate | SkillTemplate) -> TaskType:
        """Determine the task type for a template instance.

        :param template: Template instance to inspect.
        :return: Task type corresponding to the template.
        :raise TypeError: If the template type is unsupported.
        """
        if isinstance(template, SkillTemplate):
            return TaskType.SKILL
        if isinstance(template, PromptTemplate):
            return TaskType.PROMPT
        msg = f"Unsupported template type: {type(template)}"
        raise TypeError(msg)


def build_empty_registry() -> Registry:
    """Create an empty registry.

    :return: New Registry instance.
    """
    return Registry()
