"""Prompt registry.

Maps (name, version) -> prompt instance.

Users typically:
* create prompts in their project
* register them here or pass a custom registry
"""

import logging

from wags_llm.prompts.base import BasePromptTemplate

_logger = logging.getLogger(__name__)


class PromptRegistry:
    """Store and retrieve prompts."""

    def __init__(self) -> None:
        """Initialize an empty prompt registry."""
        self._prompts: dict[tuple[str, str], BasePromptTemplate] = {}

    def register(self, prompt: BasePromptTemplate) -> None:
        """Register a prompt.

        :param prompt: Prompt instance to register.
        """
        _logger.debug(
            "Registering prompt: name='%s', version='%s'", prompt.name, prompt.version
        )
        self._prompts[(prompt.name, prompt.version)] = prompt

    def get(self, name: str, version: str) -> BasePromptTemplate:
        """Retrieve a prompt.

        :param name: Prompt name.
        :param version: Prompt version.
        :return: Registered prompt.
        :raise KeyError: If prompt is not found.
        """
        try:
            return self._prompts[(name, version)]
        except KeyError as exc:
            msg = f"Prompt not found: ({name}, {version})"
            _logger.exception(msg)
            raise KeyError(msg) from exc


def build_empty_registry() -> PromptRegistry:
    """Create an empty prompt registry.

    :return: New PromptRegistry instance.
    """
    return PromptRegistry()
