"""Test that Registry works correctly"""

import re
from collections.abc import Mapping
from typing import Any

import pytest

from wags_llm.registry import Registry, build_empty_registry
from wags_llm.templates.base import PromptTemplate


class DummyPrompt(PromptTemplate):
    """Simple prompt for registry tests."""

    name = "test_task"
    version = "v1"

    def build_system_prompt(self) -> str:
        """Build the system prompt.

        :return: System prompt string.
        """
        return "Return valid JSON only."

    def build_user_prompt(self, payload: Mapping[str, Any]) -> str:
        """Build the user prompt.

        :param payload: JSON-serializable task data.

        Example:
            payload = {"text": "hello"}

        :return: User prompt string.
        """
        return f"Payload: {payload}"


def test_register_and_get_prompt():
    """Register and retrieve a prompt."""
    registry = Registry()
    prompt = DummyPrompt()

    registry.register(prompt)

    assert registry.get("test_task", "v1") is prompt


def test_build_empty_registry():
    """Test that build_empty_registry works correctly and prompt registry raises KeyError when no prompts are registered"""
    registry = build_empty_registry()
    with pytest.raises(
        KeyError, match=re.escape("'Template not found: (test_task, v1)'")
    ):
        assert registry.get("test_task", "v1")
