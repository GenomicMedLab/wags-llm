import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from wags_llm.skills.base import BaseSkillTemplate, SkillTemplateError
from wags_llm.skills.registry import SkillRegistry, build_empty_registry


class DummySkill(BaseSkillTemplate):
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
    registry = SkillRegistry()
    skill = DummySkill()

    registry.register(skill)

    assert registry.get("test_skill", "v1") is skill


def test_build_empty_registry():
    registry = build_empty_registry()
    with pytest.raises(
        KeyError, match=re.escape("'Skill not found: (test_skill, v1)'")
    ):
        assert registry.get("test_skill", "v1")


def test_invalid_skill_filename():
    """Test that an invalid skill filename raises SkillTemplateError."""

    class InvalidSkill(BaseSkillTemplate):
        skill_path = Path("tests/unit/skills/invalid.md")

        def build_user_prompt(self, payload) -> str:
            return f"Payload: {payload}"

    with pytest.raises(SkillTemplateError):
        _ = InvalidSkill().name
