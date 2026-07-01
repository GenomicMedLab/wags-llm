"""Test that BedrockClaudeJsonClient works correctly"""

from unittest.mock import patch

import pytest

from wags_llm.client.bedrock import (
    BedrockClaudeJsonClient,
    EffortLevel,
    LLMEmptyResponseError,
    LLMInvalidEffortError,
    LLMInvocationError,
    LLMJsonDecodeError,
    LLMResponseFormatError,
)

TEST_MODEL_ID = "test-model"
TEST_REGION_NAME = "us-east-1"
TEST_PROFILE_NAME = "my-profile"

TEST_SYSTEM_PROMPT = "system"
TEST_USER_PROMPT = "user"


class FakeBedrockRuntimeClient:
    """Fake Bedrock runtime client for tests."""

    def __init__(self, response=None, error=None):
        """Initialize the fake runtime client.

        :param response: Fake response returned by `converse`.
        :param error: Optional error raised by `converse`.
        """
        self.response = response
        self.error = error
        self.captured_request = None

    def converse(self, **kwargs):
        """Return a fake converse response.

        :param kwargs: Converse request arguments.
        :return: Fake response payload.
        :raise Exception: If configured with an error.
        """
        self.captured_request = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeSession:
    """Fake boto3 session for tests."""

    def __init__(self, runtime_client: FakeBedrockRuntimeClient):
        """Initialize the fake session.

        :param runtime_client: Fake client returned by `client()`.
        """
        self.runtime_client = runtime_client

    def client(self, service_name: str, region_name: str):
        """Return the fake Bedrock runtime client.

        :param service_name: Requested AWS service.
        :param region_name: AWS region name.
        :return: Fake Bedrock runtime client.
        """
        assert service_name == "bedrock-runtime"
        assert region_name == TEST_REGION_NAME
        return self.runtime_client


def test_invoke_json_with_effort():
    """Test that invoke_json includes the effort beta config when effort is set."""
    fake_runtime_client = FakeBedrockRuntimeClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {"text": '{"value": 1}'},
                    ]
                }
            }
        }
    )

    with patch(
        "wags_llm.client.bedrock.boto3.Session",
        return_value=FakeSession(fake_runtime_client),
    ):
        client = BedrockClaudeJsonClient(
            model_id=TEST_MODEL_ID,
            region_name=TEST_REGION_NAME,
            profile_name=TEST_PROFILE_NAME,
            effort=EffortLevel.MEDIUM,
        )

        client.invoke_json(
            system_prompt=TEST_SYSTEM_PROMPT,
            user_prompt=TEST_USER_PROMPT,
        )

    assert fake_runtime_client.captured_request["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
    }


def test_invoke_json_without_effort_omits_field():
    """Test that invoke_json omits additionalModelRequestFields when effort is unset."""
    fake_runtime_client = FakeBedrockRuntimeClient(
        response={"output": {"message": {"content": [{"text": '{"value": 1}'}]}}}
    )

    with patch(
        "wags_llm.client.bedrock.boto3.Session",
        return_value=FakeSession(fake_runtime_client),
    ):
        client = BedrockClaudeJsonClient(
            model_id=TEST_MODEL_ID,
            region_name=TEST_REGION_NAME,
            profile_name=TEST_PROFILE_NAME,
        )

        client.invoke_json(
            system_prompt=TEST_SYSTEM_PROMPT,
            user_prompt=TEST_USER_PROMPT,
        )

    assert "additionalModelRequestFields" not in fake_runtime_client.captured_request


def test_invalid_effort_raises():
    """Test that an invalid effort value raises LLMInvalidEffortError at construction."""
    with pytest.raises(LLMInvalidEffortError, match=r"Effort must"):
        BedrockClaudeJsonClient(
            model_id=TEST_MODEL_ID,
            region_name=TEST_REGION_NAME,
            profile_name=TEST_PROFILE_NAME,
            effort="extreme",
        )


def test_invoke_json_success():
    """Test that invoke_json works correctly"""
    fake_runtime_client = FakeBedrockRuntimeClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {"text": '{"value": 1}'},
                    ]
                }
            }
        }
    )

    with patch(
        "wags_llm.client.bedrock.boto3.Session",
        return_value=FakeSession(fake_runtime_client),
    ) as mock_session:
        client = BedrockClaudeJsonClient(
            model_id=TEST_MODEL_ID,
            region_name=TEST_REGION_NAME,
            profile_name=TEST_PROFILE_NAME,
        )

        invoke_json_response = client.invoke_json(
            system_prompt=TEST_SYSTEM_PROMPT,
            user_prompt=TEST_USER_PROMPT,
        )

    mock_session.assert_called_once_with(profile_name=TEST_PROFILE_NAME)
    assert invoke_json_response.parsed_json == {"value": 1}
    assert invoke_json_response.raw_text == '{"value": 1}'


def test_invoke_json_with_json_schema():
    """Test that invoke_json passes structured output config to Bedrock."""
    fake_runtime_client = FakeBedrockRuntimeClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {"text": '{"value": 1}'},
                    ]
                }
            }
        }
    )

    json_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "integer"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    with patch(
        "wags_llm.client.bedrock.boto3.Session",
        return_value=FakeSession(fake_runtime_client),
    ) as mock_session:
        client = BedrockClaudeJsonClient(
            model_id=TEST_MODEL_ID,
            region_name=TEST_REGION_NAME,
            profile_name=TEST_PROFILE_NAME,
        )

        invoke_json_response = client.invoke_json(
            system_prompt=TEST_SYSTEM_PROMPT,
            user_prompt=TEST_USER_PROMPT,
            json_schema=json_schema,
        )

    mock_session.assert_called_once_with(profile_name=TEST_PROFILE_NAME)

    assert invoke_json_response.parsed_json == {"value": 1}
    assert invoke_json_response.raw_text == '{"value": 1}'


def test_invoke_json_custom_profile():
    """Test that invoke_json uses the provided AWS profile"""
    fake_runtime_client = FakeBedrockRuntimeClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {"text": '{"value": 1}'},
                    ]
                }
            }
        }
    )

    with patch(
        "wags_llm.client.bedrock.boto3.Session",
        return_value=FakeSession(fake_runtime_client),
    ) as mock_session:
        client = BedrockClaudeJsonClient(
            model_id=TEST_MODEL_ID,
            region_name=TEST_REGION_NAME,
            profile_name="lab-profile",
        )

        client.invoke_json(
            system_prompt=TEST_SYSTEM_PROMPT,
            user_prompt=TEST_USER_PROMPT,
        )

    mock_session.assert_called_once_with(profile_name="lab-profile")


def test_invoke_json_invalid_json():
    """Test that invoke_json raises error when client returns non-json response"""
    fake_runtime_client = FakeBedrockRuntimeClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {"text": "not json"},
                    ]
                }
            }
        }
    )

    with patch(
        "wags_llm.client.bedrock.boto3.Session",
        return_value=FakeSession(fake_runtime_client),
    ):
        client = BedrockClaudeJsonClient(
            model_id=TEST_MODEL_ID,
            region_name=TEST_REGION_NAME,
            profile_name=TEST_PROFILE_NAME,
        )

        with pytest.raises(
            LLMJsonDecodeError, match=r"Model returned non-JSON output:"
        ):
            client.invoke_json(
                system_prompt=TEST_SYSTEM_PROMPT,
                user_prompt=TEST_USER_PROMPT,
            )


def test_invoke_json_empty_output():
    """Test that invoke_json raises error when client returns empty text"""
    fake_runtime_client = FakeBedrockRuntimeClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {"text": "   "},
                    ]
                }
            }
        }
    )

    with patch(
        "wags_llm.client.bedrock.boto3.Session",
        return_value=FakeSession(fake_runtime_client),
    ):
        client = BedrockClaudeJsonClient(
            model_id=TEST_MODEL_ID,
            region_name=TEST_REGION_NAME,
            profile_name=TEST_PROFILE_NAME,
        )

        with pytest.raises(
            LLMEmptyResponseError, match=r"Model returned empty output\."
        ):
            client.invoke_json(
                system_prompt=TEST_SYSTEM_PROMPT,
                user_prompt=TEST_USER_PROMPT,
            )


def test_invoke_json_bad_response_format():
    """Test that invoke_json raises error when client returns unexpected structure"""
    fake_runtime_client = FakeBedrockRuntimeClient(response={"bad": "response"})

    with patch(
        "wags_llm.client.bedrock.boto3.Session",
        return_value=FakeSession(fake_runtime_client),
    ):
        client = BedrockClaudeJsonClient(
            model_id=TEST_MODEL_ID,
            region_name=TEST_REGION_NAME,
            profile_name=TEST_PROFILE_NAME,
        )

        with pytest.raises(
            LLMResponseFormatError,
            match=r"Unexpected Converse response structure, missing key:",
        ):
            client.invoke_json(
                system_prompt=TEST_SYSTEM_PROMPT,
                user_prompt=TEST_USER_PROMPT,
            )


def test_invoke_json_no_text_content():
    """Test that invoke_json raises error when client returns no text output"""
    fake_runtime_client = FakeBedrockRuntimeClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {"type": "text"},
                    ]
                }
            }
        }
    )

    with patch(
        "wags_llm.client.bedrock.boto3.Session",
        return_value=FakeSession(fake_runtime_client),
    ):
        client = BedrockClaudeJsonClient(
            model_id=TEST_MODEL_ID,
            region_name=TEST_REGION_NAME,
            profile_name=TEST_PROFILE_NAME,
        )

        with pytest.raises(
            LLMResponseFormatError,
            match=r"No text content found in Converse response:",
        ):
            client.invoke_json(
                system_prompt=TEST_SYSTEM_PROMPT,
                user_prompt=TEST_USER_PROMPT,
            )


def test_invoke_json_converse_error():
    """Test that invoke_json raises error when client fails"""
    fake_runtime_client = FakeBedrockRuntimeClient(error=RuntimeError("ope"))

    with patch(
        "wags_llm.client.bedrock.boto3.Session",
        return_value=FakeSession(fake_runtime_client),
    ):
        client = BedrockClaudeJsonClient(
            model_id=TEST_MODEL_ID,
            region_name=TEST_REGION_NAME,
            profile_name=TEST_PROFILE_NAME,
        )

        with pytest.raises(LLMInvocationError, match=r"Bedrock converse failed: ope"):
            client.invoke_json(
                system_prompt=TEST_SYSTEM_PROMPT,
                user_prompt=TEST_USER_PROMPT,
            )
