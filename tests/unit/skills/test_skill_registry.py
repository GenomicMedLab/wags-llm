import re
from collections.abc import Mapping
from typing import Any

import pytest

from wags_llm.skills.base import BaseSkillTemplate
from wags_llm.skills.registry import SkillRegistry, build_empty_registry


class DummySkill(BaseSkillTemplate):
    skill_path = "skills/entity_detection.md"
    version = "v1"

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

    assert registry.get("entity_detection", "v1") is skill


def test_build_empty_registry():
    registry = build_empty_registry()
    with pytest.raises(
        KeyError, match=re.escape("'Skill not found: (entity_detection, v1)'")
    ):
        assert registry.get("entity_detection", "v1")
