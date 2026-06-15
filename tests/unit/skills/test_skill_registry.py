import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from wags_llm.registry import Registry, build_empty_registry
from wags_llm.templates.skill_template import SkillTemplate, SkillTemplateError


class DummySkill(SkillTemplate):
    skill_path = Path("tests/unit/skills/test_skill_v1.md")

    def build_user_prompt(self, payload: Mapping[str, Any]) -> str:
        """Build the user prompt.

        :param payload: JSON-serializable task data.

        Example:
            payload = {"text": "hello"}

        :return: User prompt string.
        """
        return f"Payload: {payload}"


def test_register_and_get_skill():
    registry = Registry()
    skill = DummySkill()

    registry.register(skill)

    assert registry.get("test_skill", "v1") is skill


def test_build_empty_registry():
    registry = build_empty_registry()
    with pytest.raises(
        KeyError, match=re.escape("'Template not found: (test_skill, v1)'")
    ):
        assert registry.get("test_skill", "v1")


def test_invalid_skill_filename():
    """Test that an invalid skill filename raises SkillTemplateError."""

    class InvalidSkill(SkillTemplate):
        skill_path = Path("tests/unit/skills/invalid.md")

        def build_user_prompt(self, payload) -> str:
            return f"Payload: {payload}"

    with pytest.raises(SkillTemplateError):
        _ = InvalidSkill().name
