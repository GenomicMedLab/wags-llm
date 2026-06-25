"""Test that Registry works correctly"""

import re
from collections.abc import Mapping

# NEW
from pathlib import Path
from typing import Any

import pytest

from wags_llm.registry import Registry, TaskType, build_empty_registry
from wags_llm.templates.base import PromptTemplate
from wags_llm.templates.skill_template import SkillTemplate


class DummySkill(SkillTemplate):
    """Simple skill for registry tests."""

    skill_path = Path("tests/unit/prompts/test_registry_0.1.0.md")

    def build_user_prompt(self, payload: Mapping[str, Any]) -> str:
        """Build the user prompt.

        :param payload: JSON-serializable task data.

        Example:
            payload = {"text": "hello"}

        :return: User prompt string.
        """
        return f"Payload: {payload}"


class DummyPrompt(PromptTemplate):
    """Simple prompt for registry tests."""

    name = "test_registry"
    version = "0.1.0"

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

    assert registry.get("test_registry", "0.1.0", TaskType.PROMPT) is prompt


def test_build_empty_registry():
    """Test that build_empty_registry works correctly and prompt registry raises KeyError when no prompts are registered"""
    registry = build_empty_registry()
    with pytest.raises(
        KeyError,
        match=re.escape("'Template not found: (test_registry, 0.1.0, prompt)'"),
    ):
        assert registry.get("test_registry", "0.1.0", TaskType.PROMPT)


def test_prompt_and_skill_can_share_name_and_version():
    """Register prompt and skill with same name/version."""
    registry = Registry()

    prompt = DummyPrompt()
    skill = DummySkill()

    registry.register(prompt)
    registry.register(skill)

    assert registry.get("test_registry", "0.1.0", TaskType.PROMPT) is prompt
    assert registry.get("test_registry", "0.1.0", TaskType.SKILL) is skill


def test_registering_duplicate_skill_raises_value_error():
    """Raise ValueError when registering duplicate skill name/version."""
    registry = Registry()
    skill = DummySkill()

    registry.register(skill)

    with pytest.raises(
        ValueError,
        match=re.escape("Template already registered:(test_registry, 0.1.0, skill)"),
    ):
        registry.register(DummySkill())
