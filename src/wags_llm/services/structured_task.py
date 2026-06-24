"""Run LLM prompts or skills and return schema-validated structured outputs.

Inputs:
- prompt or skill name and version
- context + payload
- response model (Pydantic)

Returns validated output.

Users extend by:
- writing prompts or defining skills
- defining response models (Pydantic)
"""

import hashlib
import json
import logging
from collections.abc import Mapping
from enum import Enum
from os import getenv
from typing import Any

from pydantic import BaseModel, ValidationError

from wags_llm.cache.base import BaseCache
from wags_llm.client.base import LLMJsonClient
from wags_llm.client.exceptions import LLMClientError
from wags_llm.registry import Registry, build_empty_registry

_logger = logging.getLogger(__name__)

MAX_LOG_CHARS = int(getenv("MAX_LOG_CHARS", "500"))


class TaskKind(Enum):
    """Enum for task types supported by StructuredTaskRunner."""

    SKILL = "Skill"
    PROMPT = "Prompt"


class CacheLookupResult(BaseModel):
    """Result of a cache lookup.

    :var cache_key: The cache key for this request. None if caching is disabled.
    :var cached: The validated cached result. None if no cached result was found.
    """

    cache_key: str | None
    cached: BaseModel | None


class StructuredTaskRunner:
    """Run structured LLM tasks."""

    def __init__(
        self,
        client: LLMJsonClient,
        registry: Registry | None = None,
        cache: BaseCache | None = None,
    ) -> None:
        """Initialize the structured task runner.

        :param client: LLM client used to execute prompts or skills.
        :param registry: Registry used to resolve prompts or skills.
        :param cache: Optional cache for storing and retrieving task results.
        """
        self.client = client
        self.registry = registry or build_empty_registry()
        self.cache = cache

    def execute_skill(
        self,
        skill_name: str,
        skill_version: str,
        payload: Mapping[str, Any],
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Execute a skill and return validated output.

        :param skill_name: Registered skill name.
        :param skill_version: Registered skill version.
        :param payload: JSON-serializable task data.
        :param response_model: Pydantic model for validation.
        :return: Validated skill result.
        :raise RuntimeError: If execution or validation fails.
        """
        return self._execute(
            name=skill_name,
            version=skill_version,
            payload=payload,
            response_model=response_model,
            kind=TaskKind.SKILL,
        )

    def execute_prompt(
        self,
        prompt_name: str,
        prompt_version: str,
        payload: Mapping[str, Any],
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Execute a prompt and return validated output.

        :param prompt_name: Registered prompt name.
        :param prompt_version: Registered prompt version.
        :param payload: JSON-serializable task data.
        :param response_model: Pydantic model for validation.
        :return: Validated prompt result.
        :raise RuntimeError: If execution or validation fails.
        """
        return self._execute(
            name=prompt_name,
            version=prompt_version,
            payload=payload,
            response_model=response_model,
            kind=TaskKind.PROMPT,
        )

    def _execute(
        self,
        name: str,
        version: str,
        payload: Mapping[str, Any],
        response_model: type[BaseModel],
        kind: TaskKind,
    ) -> BaseModel:
        """Execute a task and return validated output.

        :param name: Registered task name.
        :param version: Registered task version.
        :param payload: JSON-serializable task data.
        :param response_model: Pydantic model for validation.
        :param kind: Display label for the registered task type, either "Skill" or "Prompt".
        :return: Validated task result.
        :raise RuntimeError: If execution or validation fails.
        """
        registered_task = self.registry.get(name, version)

        cache_result = self._check_cache(
            name=name,
            version=version,
            payload=payload,
            response_model=response_model,
        )
        if cache_result.cached is not None:
            return cache_result.cached

        try:
            invoke_json_response = self.client.invoke_json(
                system_prompt=registered_task.build_system_prompt(),
                user_prompt=registered_task.build_user_prompt(payload=payload),
                json_schema=response_model.model_json_schema(),
            )

            result = response_model.model_validate(invoke_json_response.parsed_json)

            if self.cache is not None and cache_result.cache_key is not None:
                self.cache.set(cache_result.cache_key, result.model_dump())

        except (LLMClientError, ValidationError) as exc:
            msg = f"{kind.value} execution failed for {name} version {version}: {exc}"
            _logger.exception(msg)
            raise RuntimeError(msg) from exc
        else:
            return result

    def _cache_key(
        self,
        name: str,
        version: str,
        payload: Mapping[str, Any],
    ) -> str:
        """Build a cache key for a task run.

        :param name: Registered name.
        :param version: Registered version.
        :param payload: JSON-serializable task data.
        :return: Stable cache key.
        """
        cache_payload = {
            "payload": dict(payload),
            "model": self.client.model_id,
            "name": name,
            "version": version,
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

    def _check_cache(
        self,
        name: str,
        version: str,
        payload: Mapping[str, Any],
        response_model: type[BaseModel],
    ) -> CacheLookupResult:
        """Check cache for an existing result.

        :param name: Registered name.
        :param version: Registered version.
        :param payload: JSON-serializable task data.
        :param response_model: Pydantic model for validation.
        :return: Container with the cache key and cached object (validated result). Cached object is None on cache miss.
        """
        if self.cache is not None:
            cache_key = self._cache_key(
                name=name,
                version=version,
                payload=payload,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return CacheLookupResult(
                    cache_key=cache_key, cached=response_model.model_validate(cached)
                )
        else:
            cache_key = None

        return CacheLookupResult(cache_key=cache_key, cached=None)
