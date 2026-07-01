"""Base template interface.

Users extend this to define new tasks.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class TemplateType(StrEnum):
    """Enum for template types supported by StructuredTaskRunner."""

    SKILL = "skill"
    PROMPT = "prompt"


class BaseTemplate(ABC):
    """Base template.

    :var name: Template name.
    :var version: Template version.
    """

    name: str
    version: str

    @abstractmethod
    def build_system_prompt(self) -> str:
        """Build the system prompt.

        Defines global instructions for the task (e.g., output format, constraints).

        :return: System prompt string.
        """

    @abstractmethod
    def build_user_prompt(self, payload: Mapping[str, Any]) -> str:
        """Build the user prompt.

        :param payload: JSON-serializable task data used to construct the prompt.
        :return: User prompt string.
        """
