"""Bedrock client.

* Handles model invocation and response parsing.
* Used by services. Not typically modified by users.
"""

import json
import logging
from typing import Any

import boto3

from wags_llm.client.base import InvokeJsonResponse, LLMJsonClient
from wags_llm.client.exceptions import (
    LLMEmptyResponseError,
    LLMInvocationError,
    LLMJsonDecodeError,
    LLMResponseFormatError,
)

_logger = logging.getLogger(__name__)


class BedrockClaudeJsonClient(LLMJsonClient):
    """Bedrock Claude Converse JSON client."""

    def __init__(
        self,
        model_id: str,
        region_name: str,
        profile_name: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> None:
        """Initialize the Bedrock Claude client.

        :param model_id: Bedrock model identifier.
        :param region_name: AWS region for the Bedrock runtime client.
        :param profile_name: AWS profile name.
        :param max_tokens: Maximum number of tokens to request from the model.
        :param temperature: Sampling temperature.
        """
        _logger.debug(
            "BedrockClaudeJsonClient config: model_id='%s', region_name='%s', profile_name='%s', max_tokens=%i, temperature=%f",
            model_id,
            region_name,
            profile_name,
            max_tokens,
            temperature,
        )
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        session = boto3.Session(profile_name=profile_name)
        self._client = session.client("bedrock-runtime", region_name=region_name)
        _logger.info(
            "BedrockClaudeJsonClient successfully initialized for model_id='%s'",
            model_id,
        )

    def invoke_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> InvokeJsonResponse:
        """Call the Bedrock Converse API and return response containing structured JSON
        and raw text.

        When a JSON schema is provided, it is sent to Bedrock using structured output
        configuration to constrain the model response format. Without a JSON schema,
        the model may return JSON along with additional reasoning or explanatory text.

        Raw text is returned for audit or debugging purposes.

        :param system_prompt: System prompt text.
        :param user_prompt: User prompt text.
        :param json_schema: Optional JSON schema used to constrain the model response.
        :return: InvokeJsonResponse containing parsed_json and raw text.
        :raise LLMInvocationError: If the Bedrock call fails.
        :raise LLMResponseFormatError: If the response shape is invalid.
        :raise LLMEmptyResponseError: If the model returns empty text.
        :raise LLMJsonDecodeError: If the model output is not valid JSON.
        """
        converse_params: dict[str, Any] = {
            "modelId": self.model_id,
            "system": [{"text": system_prompt}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": user_prompt}],
                }
            ],
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        if json_schema:
            converse_params["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "schema": json.dumps(json_schema),
                            "name": "structured_response",
                        }
                    },
                }
            }

        try:
            response = self._client.converse(**converse_params)
        except Exception as exc:
            msg = f"Bedrock converse failed: {exc}"
            _logger.exception(msg)
            raise LLMInvocationError(msg) from exc

        raw_text = self._extract_text_from_converse_response(response)

        if not raw_text.strip():
            msg = "Model returned empty output."
            _logger.exception(msg)
            raise LLMEmptyResponseError(msg)

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            msg = f"Model returned non-JSON output: {exc}; output={raw_text!r}"
            _logger.exception(msg)
            raise LLMJsonDecodeError(msg) from exc

        return InvokeJsonResponse(parsed_json=parsed, raw_text=raw_text)

    def _extract_text_from_converse_response(self, response: dict[str, Any]) -> str:
        """Extract text content from a Bedrock Converse response.

        :param response: Raw response object from converse.
        :return: Joined response text.
        :raise LLMResponseFormatError: If the response does not contain expected keys
            or text blocks.
        """
        _logger.debug("Bedrock Claude usage=%s", response.get("usage", {}))
        _logger.debug("Bedrock Claude metrics=%s", response.get("metrics", {}))

        try:
            content = response["output"]["message"]["content"]
        except KeyError as exc:
            msg = f"Unexpected Converse response structure, missing key: {exc}; response={response}"
            _logger.exception(msg)
            raise LLMResponseFormatError(msg) from exc

        _logger.debug("Bedrock Claude content=%s", content)

        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and "text" in item
        ]

        if not text_parts:
            msg = f"No text content found in Converse response: {response}"
            _logger.exception(msg)
            raise LLMResponseFormatError(msg)

        return "\n".join(part.strip() for part in text_parts if part.strip()).strip()
