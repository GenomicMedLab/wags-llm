"""Base client interfaces for LLMs.

This module defines the protocol used by higher-level services to invoke
LLMs for structured JSON tasks without depending on a specific provider
implementation.
"""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class InvokeJsonResponse(BaseModel):
    """Define expected response model for `invoke_json`"""

    model_config = ConfigDict(extra="allow")

    parsed_json: dict[str, Any]
    raw_text: str


@runtime_checkable
class LLMJsonClient(Protocol):
    """Protocol for LLM clients that return structured JSON responses."""

    model_id: str

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the client implementation.

        Concrete clients may accept provider-specific configuration.
        """

    def invoke_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> InvokeJsonResponse:
        """Invoke the model and return response containing parsed JSON and raw text.

        Raw text is returned for audit or debugging purposes.

        :param system_prompt: System prompt text.
        :param user_prompt: User prompt text.
        :returns: Structured response from the client.
        :raise LLMInvocationError: If the invocation call fails.
        :raise LLMResponseFormatError: If the response shape is invalid.
        :raise LLMEmptyResponseError: If the model returns empty text.
        :raise LLMJsonDecodeError: If the model output is not valid JSON.
        """
        ...
