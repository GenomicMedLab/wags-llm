"""LLM custom exceptions"""


class LLMClientError(Exception):
    """Base client error."""


class LLMInvocationError(LLMClientError):
    """Raised when the invocation call fails."""


class LLMResponseFormatError(LLMClientError):
    """Raised when the response shape is invalid."""


class LLMEmptyResponseError(LLMClientError):
    """Raised when the model returns empty text."""


class LLMJsonDecodeError(LLMClientError):
    """Raised when the model output is not valid JSON."""
