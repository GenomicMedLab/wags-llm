"""Test that StructuredTaskRunner works correctly"""

from typing import Any

import pytest
from pydantic import BaseModel

from wags_llm.cache.in_memory import InMemoryCache
from wags_llm.client.base import InvokeJsonResponse, LLMJsonClient
from wags_llm.registry import Registry
from wags_llm.services.structured_task import StructuredTaskRunner
from wags_llm.templates.base import PromptTemplate


class DummyPrompt(PromptTemplate):
    """Simple prompt for service tests."""

    name = "test_task"
    version = "v1"

    def build_system_prompt(self) -> str:
        """Build the system prompt."""
        return "Return valid JSON only."

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


def test_run_success():
    """Test that run method works correctly"""
    registry = Registry()
    registry.register(DummyPrompt())

    service = StructuredTaskRunner(
        client=DummyClient(),
        registry=registry,
    )

    result = service.execute_prompt(
        prompt_name="test_task",
        prompt_version="v1",
        payload={"text": "hello"},
        response_model=ResultModel,
    )

    assert result.value == 1


def test_run_uses_cache():
    """Test that run method works correctly with cache"""
    registry = Registry()
    registry.register(DummyPrompt())
    client = DummyClient()
    cache = InMemoryCache()

    service = StructuredTaskRunner(
        client=client,
        registry=registry,
        cache=cache,
    )

    result1 = service.execute_prompt(
        prompt_name="test_task",
        prompt_version="v1",
        payload={"x": 1},
        response_model=ResultModel,
    )
    result2 = service.execute_prompt(
        prompt_name="test_task",
        prompt_version="v1",
        payload={"x": 1},
        response_model=ResultModel,
    )

    assert result1.value == 1
    assert result2.value == 1
    assert client.calls == 1


def test_run_cache_miss_for_different_payload():
    """Test that run method works correctly with cache that uses different payload"""
    registry = Registry()
    registry.register(DummyPrompt())
    client = DummyClient()
    cache = InMemoryCache()

    service = StructuredTaskRunner(
        client=client,
        registry=registry,
        cache=cache,
    )

    service.execute_prompt(
        prompt_name="test_task",
        prompt_version="v1",
        payload={"x": 1},
        response_model=ResultModel,
    )
    service.execute_prompt(
        prompt_name="test_task",
        prompt_version="v1",
        payload={"x": 2},
        response_model=ResultModel,
    )

    assert client.calls == 2


def test_run_validation_error():
    """Test that run raises error when response validation fails."""
    registry = Registry()
    registry.register(DummyPrompt())

    service = StructuredTaskRunner(
        client=BadClient(),
        registry=registry,
    )

    with pytest.raises(RuntimeError, match="prompt execution failed"):
        service.execute_prompt(
            prompt_name="test_task",
            prompt_version="v1",
            payload={"text": "hello"},
            response_model=ResultModel,
        )
