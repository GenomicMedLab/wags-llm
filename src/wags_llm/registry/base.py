"""Registry.

Maps (name, version, TemplateType) -> template instance.
Template instances can be either prompts or skills.

Users typically:
* create prompts or skills in their project
* register them here or pass a custom registry
"""

import logging
from types import MappingProxyType

from wags_llm.templates.base import TemplateType
from wags_llm.templates.prompt_template import PromptTemplate
from wags_llm.templates.skill_template import SkillTemplate

_logger = logging.getLogger(__name__)

_TEMPLATE_CLASS_TO_TYPE = MappingProxyType(
    {
        SkillTemplate: TemplateType.SKILL,
        PromptTemplate: TemplateType.PROMPT,
    }
)


class Registry:
    """Store and retrieve prompt and skill templates."""

    def __init__(self) -> None:
        """Initialize an empty template registry."""
        self._templates: dict[
            tuple[str, str, TemplateType], PromptTemplate | SkillTemplate
        ] = {}

    def register(self, template: PromptTemplate | SkillTemplate) -> None:
        """Register a template.

        :param template: Template instance to register.
        :raise TypeError: If the template type is unsupported.
        :raise ValueError: If a template with the same name, version, and template type is already registered.
        """
        for cls, mapped_type in _TEMPLATE_CLASS_TO_TYPE.items():
            if isinstance(template, cls):
                template_type = mapped_type
                break
        else:
            msg = f"Unsupported template type: {type(template)}"
            raise TypeError(msg)

        key = (template.name, template.version, template_type)

        _logger.debug(
            "Registering template: name='%s', version='%s', template_type='%s'",
            template.name,
            template.version,
            template_type.value,
        )

        if key in self._templates:
            msg = f"Template already registered:({template.name}, {template.version}, {template_type.value})"
            _logger.error(msg)
            raise ValueError(msg)

        self._templates[key] = template

    def get(
        self,
        name: str,
        version: str,
        template_type: TemplateType,
    ) -> PromptTemplate | SkillTemplate:
        """Retrieve a template by name, version, and template type.

        :param name: Template name.
        :param version: Template version.
        :param template_type: Template type.
        :return: Registered template.
        :raise KeyError: If template not found.
        """
        key = (name, version, template_type)

        try:
            return self._templates[key]
        except KeyError as exc:
            msg = f"Template not found: ({name}, {version}, {template_type.value})"
            _logger.exception(msg)
            raise KeyError(msg) from exc


def build_empty_registry() -> Registry:
    """Create an empty registry.

    :return: New Registry instance.
    """
    return Registry()
