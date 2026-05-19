import pytest
from unittest.mock import MagicMock

from wags_llm.client.bedrock import BedrockClaudeJsonClient
from wags_llm.client.exceptions import LLMJsonDecodeError


def make_client():
    client = BedrockClaudeJsonClient(
        model_id="test-model",
        region_name="us-east-1",
        profile_name="test-profile",
    )

    # override real AWS client
    client._client = MagicMock()

    return client


def test_invoke_json_clean():
    client = make_client()

    client._client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {"text": '{"evidence_line_type_direction": "unclear"}'}
                ]
            }
        }
    }

    result = client.invoke_json("sys", "user")

    assert result.parsed_json["evidence_line_type_direction"] == "unclear"
    assert result.raw_text == '{"evidence_line_type_direction": "unclear"}'


def test_invoke_json_extracts_json_from_text():
    client = make_client()

    client._client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "text": """
Here is the analysis:

This evidence is unclear.

{"evidence_line_type_direction": "unclear"}

Done.
"""
                    }
                ]
            }
        }
    }

    result = client.invoke_json("sys", "user")

    assert result.parsed_json["evidence_line_type_direction"] == "unclear"


def test_invoke_json_fails_when_no_json():
    client = make_client()

    client._client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {"text": "This is just free text with no structured output."}
                ]
            }
        }
    }

    with pytest.raises(LLMJsonDecodeError):
        client.invoke_json("sys", "user")


def test_invoke_json_fails_on_broken_json():
    client = make_client()

    client._client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {"text": '{"evidence_line_type_direction": "unclear"'}  # missing }
                ]
            }
        }
    }

    with pytest.raises(LLMJsonDecodeError):
        client.invoke_json("sys", "user")