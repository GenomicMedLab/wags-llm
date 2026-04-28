"""Run LLM prompts and return schema-validated structured outputs.

Inputs:
- prompt
- context + payload
- response model (Pydantic)

Returns validated output.

Users extend by:
- writing prompts
- defining response models (Pydantic)
"""

import hashlib
import json
import logging
from collections.abc import Mapping
from os import getenv
from typing import Any

from pydantic import BaseModel, ValidationError

from wags_llm.cache.base import BaseCache
from wags_llm.client.base import LLMJsonClient
from wags_llm.client.exceptions import LLMClientError
from wags_llm.prompts.registry import PromptRegistry, build_empty_registry

_logger = logging.getLogger(__name__)

MAX_LOG_CHARS = int(getenv("MAX_LOG_CHARS", "500"))


class StructuredTaskRunner:
    """Run structured LLM tasks."""

    def __init__(
        self,
        client: LLMJsonClient,
        prompt_registry: PromptRegistry | None = None,
        cache: BaseCache | None = None,
    ) -> None:
        """Initialize the structured task runner.

        :param client: LLM client used to execute prompts.
        :param prompt_registry: Registry used to resolve prompts.
        :param cache: Optional cache for storing and retrieving task results.
        """
        self.client = client
        self.prompt_registry = prompt_registry or build_empty_registry()
        self.cache = cache

    def execute(
        self,
        prompt_name: str,
        prompt_version: str,
        payload: Mapping[str, Any],
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Execute a task and return validated output.

        :param prompt_name: Registered prompt name.
        :param prompt_version: Registered prompt version.
        :param payload: JSON-serializable task data.
        :param response_model: Pydantic model for validation.
        :return: Validated task result.
        :raise RuntimeError: If execution or validation fails.
        """
        prompt = self.prompt_registry.get(prompt_name, prompt_version)

        if self.cache is not None:
            cache_key = self._cache_key(
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                payload=payload,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return response_model.model_validate(cached)
        else:
            cache_key = None

        try:
            invoke_json_response = self.client.invoke_json(
                system_prompt=prompt.build_system_prompt(),
                user_prompt=prompt.build_user_prompt(payload=payload),
            )

            result = response_model.model_validate(invoke_json_response.parsed_json)

            if self.cache is not None and cache_key is not None:
                self.cache.set(cache_key, result.model_dump())

        except (LLMClientError, ValidationError) as exc:
            msg = f"Task failed: {exc}"
            _logger.exception(msg)
            raise RuntimeError(msg) from exc
        else:
            return result

    def _cache_key(
        self,
        prompt_name: str,
        prompt_version: str,
        payload: Mapping[str, Any],
    ) -> str:
        """Build a cache key for a task run.

        :param prompt_name: Registered prompt name.
        :param prompt_version: Registered prompt version.
        :param payload: JSON-serializable task data.
        :return: Stable cache key.
        """
        cache_payload = {
            "payload": dict(payload),
            "model": self.client.model_id,
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
        }
        normalized = json.dumps(cache_payload, sort_keys=True, separators=(",", ":"))
        cache_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        # Truncate to 500 chars to keep log clean
        cache_ctx = (
            (str(cache_payload)[:500] + "...")
            if len(str(cache_payload)) > MAX_LOG_CHARS
            else cache_payload
        )
        _logger.debug(
            "Cache lookup using key='%s' (for cache_payload=%s)", cache_key, cache_ctx
        )
        return cache_key
