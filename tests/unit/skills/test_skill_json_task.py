"""Test that StructuredTaskRunner works correctly for skills"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from wags_llm.cache.in_memory import InMemoryCache
from wags_llm.client.base import InvokeJsonResponse, LLMJsonClient
from wags_llm.registry import Registry
from wags_llm.services.structured_task import StructuredTaskRunner
from wags_llm.templates.skill_template import SkillTemplate, SkillTemplateError


class DummySkill(SkillTemplate):
    """Simple skill for service tests."""

    skill_path = Path("tests/unit/skills/test_skill_v1.md")

    def build_user_prompt(self, payload) -> str:
        """Build the user prompt."""
        return f"Payload: {payload}"


class MissingFileSkill(SkillTemplate):
    """Missing skill file for service tests."""

    skill_path = Path("tests/unit/skills/does_not_exist_v1.md")

    def build_user_prompt(self, payload) -> str:
        """Build the user prompt."""
        return f"Payload: {payload}"


class DummyClient(LLMJsonClient):
    """Fake client that returns a valid response."""

    model_id = "dummy-model"

    def __init__(self):
        """Initialize the fake client."""
        self.calls = 0

    def invoke_json(
        self,
        system_prompt: str,  # noqa: ARG002
        user_prompt: str,  # noqa: ARG002
        json_schema: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> InvokeJsonResponse:
        """Return a fixed JSON response."""
        self.calls += 1
        return InvokeJsonResponse(
            parsed_json={"value": 1},
            raw_text='{"value": 1}',
        )


class BadClient:
    """Fake client that returns an invalid response shape."""

    model_id = "dummy-model"

    def __init__(self):
        """Initialize the fake client."""
        self.calls = 0

    def invoke_json(
        self,
        system_prompt: str,  # noqa: ARG002
        user_prompt: str,  # noqa: ARG002
        json_schema: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> InvokeJsonResponse:
        """Return an invalid JSON shape."""
        self.calls += 1
        return InvokeJsonResponse(
            parsed_json={"wrong": "shape"},
            raw_text='{"wrong": "shape"}',
        )


class ResultModel(BaseModel):
    """Response model for service tests."""

    value: int


def test_execute_skill_success():
    """Test that execute_skill works correctly."""
    registry = Registry()
    registry.register(DummySkill())

    service = StructuredTaskRunner(
        client=DummyClient(),
        prompt_registry=registry,
    )

    result = service.execute_skill(
        skill_name="test_skill",
        skill_version="v1",
        payload={"text": "hello"},
        response_model=ResultModel,
    )

    assert result.value == 1


def test_execute_skill_file_not_found():
    """Test that execute_skill raises FileNotFoundError when skill file does not exist."""

    registry = Registry()
    registry.register(MissingFileSkill())

    service = StructuredTaskRunner(
        client=DummyClient(),
        prompt_registry=registry,
    )

    with pytest.raises(SkillTemplateError):
        service.execute_skill(
            skill_name="does_not_exist",
            skill_version="v1",
            payload={"text": "hello"},
            response_model=ResultModel,
        )


def test_execute_skill_uses_cache():
    """Test that execute_skill works correctly with cache."""
    registry = Registry()
    registry.register(DummySkill())
    client = DummyClient()
    cache = InMemoryCache()

    service = StructuredTaskRunner(
        client=client,
        prompt_registry=registry,
        cache=cache,
    )

    result1 = service.execute_skill(
        skill_name="test_skill",
        skill_version="v1",
        payload={"x": 1},
        response_model=ResultModel,
    )
    result2 = service.execute_skill(
        skill_name="test_skill",
        skill_version="v1",
        payload={"x": 1},
        response_model=ResultModel,
    )

    assert result1.value == 1
    assert result2.value == 1
    assert client.calls == 1


def test_execute_skill_cache_miss_for_different_payload():
    """Test that execute_skill cache misses on different payload."""
    registry = Registry()
    registry.register(DummySkill())
    client = DummyClient()
    cache = InMemoryCache()

    service = StructuredTaskRunner(
        client=client,
        prompt_registry=registry,
        cache=cache,
    )

    service.execute_skill(
        skill_name="test_skill",
        skill_version="v1",
        payload={"x": 1},
        response_model=ResultModel,
    )
    service.execute_skill(
        skill_name="test_skill",
        skill_version="v1",
        payload={"x": 2},
        response_model=ResultModel,
    )

    assert client.calls == 2


def test_execute_skill_validation_error():
    """Test that execute_skill raises RuntimeError when response validation fails."""
    registry = Registry()
    registry.register(DummySkill())

    service = StructuredTaskRunner(
        client=BadClient(),
        prompt_registry=registry,
    )

    with pytest.raises(RuntimeError, match="Task failed"):
        service.execute_skill(
            skill_name="test_skill",
            skill_version="v1",
            payload={"text": "hello"},
            response_model=ResultModel,
        )
