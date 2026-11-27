"""Offline tests: they mock HTTP and check exactly what is sent to Scaleway.

The code tests use small model schemas defined here, so they don't depend on the
model files in models/, which scripts/sync_models.py regenerates. Those files
are checked separately by test_model_files_are_valid.
"""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import yaml
from dify_plugin.entities.model import AIModelEntity, FetchFrom, ModelPropertyKey
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessageTool,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin.interfaces.model.openai_compatible.llm import OAICompatLargeLanguageModel

from models.common import build_endpoint_url
from models.llm.llm import ScalewayLargeLanguageModel
from models.speech2text.speech2text import ScalewaySpeech2TextModel
from models.text_embedding.text_embedding import ScalewayTextEmbeddingModel
from provider.scaleway import ScalewayModelProvider

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASE = "https://api.scaleway.ai/v1"
PROVIDER_CREDS = {"api_key": "scw-secret"}  # what shipped models receive

LLM_POST = "dify_plugin.interfaces.model.openai_compatible.llm.requests.post"
EMB_POST = "dify_plugin.interfaces.model.openai_compatible.text_embedding.requests.post"
STT_POST = "dify_plugin.interfaces.model.openai_compatible.speech2text.requests.post"
PROVIDER_GET = "provider.scaleway.requests.get"


# ---------- helpers ----------


def schema(data: dict) -> AIModelEntity:
    return AIModelEntity(**data, fetch_from=FetchFrom.PREDEFINED_MODEL)


CHAT_MODEL = "test-chat"
CHAT_SCHEMA = schema(
    {
        "model": CHAT_MODEL,
        "label": {"en_US": "Test chat"},
        "model_type": "llm",
        "features": ["tool-call", "multi-tool-call", "stream-tool-call", "vision", "structured-output"],
        "model_properties": {"mode": "chat", "context_size": 128000},
        "parameter_rules": [
            {"name": "temperature", "use_template": "temperature"},
            {"name": "max_tokens", "use_template": "max_tokens", "default": 4096, "min": 1, "max": 32000},
        ],
    }
)
PLAIN_MODEL = "test-plain"
PLAIN_SCHEMA = CHAT_SCHEMA.model_copy(update={"model": PLAIN_MODEL, "features": []})
EMB_MODEL = "test-embedding"
EMB_SCHEMA = schema(
    {
        "model": EMB_MODEL,
        "label": {"en_US": "Test embedding"},
        "model_type": "text-embedding",
        "model_properties": {"context_size": 32000, "max_chunks": 32},
    }
)
STT_MODEL = "test-stt"
STT_SCHEMA = schema(
    {
        "model": STT_MODEL,
        "label": {"en_US": "Test speech-to-text"},
        "model_type": "speech2text",
        "model_properties": {"file_upload_limit": 25, "supported_file_extensions": "wav,mp3"},
    }
)

TOOL = PromptMessageTool(
    name="get_weather",
    description="Get weather",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def ok_response(payload: dict) -> MagicMock:
    response = MagicMock(status_code=200)
    response.json.return_value = payload
    response.text = json.dumps(payload)
    return response


def chat_response(content: str = "ok", prompt_tokens: int = 1, completion_tokens: int = 1) -> MagicMock:
    return ok_response(
        {
            "id": "1",
            "object": "chat.completion",
            "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


def sent_body(mock_post: MagicMock) -> dict:
    return json.loads(mock_post.call_args.kwargs["data"])


def provider() -> ScalewayModelProvider:
    # ModelProvider.__init__ needs Dify internals (provider schemas, model factory)
    # that credential validation doesn't use, so skip it.
    return ScalewayModelProvider.__new__(ScalewayModelProvider)


def gpt2_estimate(self, text: str) -> int:
    # The SDK's GPT-2 tokenizer downloads a file on first use; tests don't need it.
    return len(text) // 4


# ---------- model files ----------

MODEL_FILES = sorted(p for p in MODELS_DIR.glob("*/*.yaml") if p.name != "_position.yaml")


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_model_files_are_valid(path):
    with path.open() as f:
        data = yaml.safe_load(f)
    schema(data)  # raises if the file doesn't match Dify's model schema
    assert data["model"] == path.stem


def test_position_file_lists_existing_chat_models():
    position = MODELS_DIR / "llm" / "_position.yaml"
    if not position.exists():
        pytest.skip("no _position.yaml")
    with position.open() as f:
        listed = yaml.safe_load(f) or []
    shipped = {p.stem for p in (MODELS_DIR / "llm").glob("*.yaml") if p.name != "_position.yaml"}
    assert set(listed) <= shipped


# ---------- base URL ----------


@pytest.mark.parametrize(
    ("credentials", "expected"),
    [
        ({}, "https://api.scaleway.ai/v1"),
        ({"project_id": ""}, "https://api.scaleway.ai/v1"),
        ({"project_id": "abc-123"}, "https://api.scaleway.ai/abc-123/v1"),
        ({"endpoint_url": "https://dep.ifr.fr-par.scaleway.com/v1/"}, "https://dep.ifr.fr-par.scaleway.com/v1"),
        (
            {"project_id": "abc-123", "endpoint_url": "https://dep.ifr.fr-par.scaleway.com/v1"},
            "https://dep.ifr.fr-par.scaleway.com/v1",
        ),
    ],
)
def test_build_endpoint_url(credentials, expected):
    assert build_endpoint_url(credentials) == expected


# ---------- provider ----------


@patch(PROVIDER_GET)
@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({}, f"{BASE}/models"),
        ({"project_id": " my-project "}, "https://api.scaleway.ai/my-project/v1/models"),
    ],
)
def test_provider_validation_lists_models(mock_get, extra, expected):
    mock_get.return_value = ok_response({"data": []})
    provider().validate_provider_credentials(dict(PROVIDER_CREDS, **extra))
    assert mock_get.call_args.args[0] == expected
    assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer scw-secret"


@patch(PROVIDER_GET)
def test_provider_validation_rejects_bad_key(mock_get):
    mock_get.return_value = MagicMock(status_code=401, text="unauthorized")
    with pytest.raises(CredentialsValidateFailedError, match="401"):
        provider().validate_provider_credentials(PROVIDER_CREDS)


@patch(PROVIDER_GET, side_effect=requests.ConnectionError("connection refused"))
def test_provider_validation_reports_unreachable_api(mock_get):
    with pytest.raises(CredentialsValidateFailedError, match="Could not reach"):
        provider().validate_provider_credentials(PROVIDER_CREDS)


# ---------- chat ----------


@patch(LLM_POST, return_value=chat_response("Hi", prompt_tokens=3))
def test_chat_request(mock_post):
    model = ScalewayLargeLanguageModel([CHAT_SCHEMA])
    result = model._invoke(
        CHAT_MODEL,
        dict(PROVIDER_CREDS),
        [UserPromptMessage(content="Hello")],
        {"temperature": 0.2, "frequency_penalty": 0.5, "max_tokens": 50},
        tools=[TOOL],
        stream=False,
        user="end-user-42",
    )
    body = sent_body(mock_post)
    assert mock_post.call_args.args[0] == f"{BASE}/chat/completions"
    assert body["model"] == CHAT_MODEL
    assert "user" not in body
    assert "frequency_penalty" not in body
    assert body["temperature"] == 0.2
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert body["tool_choice"] == "auto"
    assert result.message.content == "Hi"
    assert result.usage.prompt_tokens == 3


@patch(LLM_POST)
def test_chat_streaming_requests_and_reads_usage(mock_post):
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}},
    ]
    lines = [f"data: {json.dumps({'id': '1', 'object': 'chat.completion.chunk', 'model': 'm', **c})}" for c in chunks]
    response = MagicMock(status_code=200)
    response.iter_lines.return_value = [*lines, "data: [DONE]"]
    mock_post.return_value = response

    model = ScalewayLargeLanguageModel([CHAT_SCHEMA])
    out = list(model._invoke(CHAT_MODEL, dict(PROVIDER_CREDS), [UserPromptMessage(content="Hello")], {}, stream=True))

    body = sent_body(mock_post)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert "".join(c.delta.message.content or "" for c in out) == "Hello"
    usages = [c.delta.usage for c in out if c.delta.usage]
    assert usages
    assert usages[-1].prompt_tokens == 7
    assert usages[-1].completion_tokens == 2


@patch(LLM_POST, return_value=chat_response())
def test_reasoning_effort_and_structured_output_are_passed_through(mock_post):
    json_schema = {"schema": {"type": "object", "properties": {"city": {"type": "string"}}}, "name": "llm_response"}
    model = ScalewayLargeLanguageModel([CHAT_SCHEMA])
    model._invoke(
        CHAT_MODEL,
        dict(PROVIDER_CREDS),
        [UserPromptMessage(content="x")],
        {"reasoning_effort": "low", "response_format": "json_schema", "json_schema": json.dumps(json_schema)},
        stream=False,
    )
    body = sent_body(mock_post)
    assert body["reasoning_effort"] == "low"
    assert body["response_format"] == {"type": "json_schema", "json_schema": json_schema}
    assert "json_schema" not in body


@patch(LLM_POST, return_value=chat_response())
def test_shipped_model_without_tool_feature_gets_no_tools(mock_post):
    model = ScalewayLargeLanguageModel([PLAIN_SCHEMA])
    model._invoke(PLAIN_MODEL, dict(PROVIDER_CREDS), [UserPromptMessage(content="x")], {}, tools=[TOOL], stream=False)
    assert "tools" not in sent_body(mock_post)


@patch(LLM_POST, return_value=chat_response())
def test_mode_is_always_chat(mock_post):
    model = ScalewayLargeLanguageModel([CHAT_SCHEMA])
    creds = dict(PROVIDER_CREDS, mode="completion")
    model._invoke(CHAT_MODEL, creds, [UserPromptMessage(content="x")], {}, stream=False)
    assert mock_post.call_args.args[0] == f"{BASE}/chat/completions"


@patch(LLM_POST, return_value=chat_response())
def test_added_model_without_tool_setting_sends_no_tools(mock_post):
    creds = {"api_key": "k", "context_size": "32768", "max_tokens_to_sample": "4096"}
    model = ScalewayLargeLanguageModel([CHAT_SCHEMA])
    model._invoke("unlisted-model", creds, [UserPromptMessage(content="x")], {}, tools=[TOOL], stream=False)
    body = sent_body(mock_post)
    assert body["model"] == "unlisted-model"
    assert "tools" not in body


@patch(LLM_POST, return_value=chat_response())
def test_added_model_keeps_form_choices(mock_post):
    creds = {
        "api_key": "k",
        "endpoint_url": "https://api.scaleway.ai/proj/v1",
        "context_size": "32768",
        "max_tokens_to_sample": "4096",
        "function_calling_type": "no_call",
    }
    model = ScalewayLargeLanguageModel([])
    model._invoke("new-model", creds, [UserPromptMessage(content="x")], {}, tools=[TOOL], stream=False)
    assert mock_post.call_args.args[0] == "https://api.scaleway.ai/proj/v1/chat/completions"
    assert "tools" not in sent_body(mock_post)  # the user chose "Not supported"


def test_added_model_hides_frequency_penalty():
    model = ScalewayLargeLanguageModel([])
    entity = model.get_customizable_model_schema(
        "some-new-model",
        {"api_key": "k", "context_size": "32768", "max_tokens_to_sample": "4096", "function_calling_type": "no_call"},
    )
    names = [r.name for r in entity.parameter_rules]
    assert "frequency_penalty" not in names
    assert "temperature" in names


@patch.object(ScalewayLargeLanguageModel, "_get_num_tokens_by_gpt2", gpt2_estimate)
def test_token_count_includes_tool_calls_for_shipped_model():
    call = AssistantPromptMessage.ToolCall(
        id="c1",
        type="function",
        function=AssistantPromptMessage.ToolCall.ToolCallFunction(name="get_weather", arguments='{"city":"Paris"}'),
    )
    messages = [
        UserPromptMessage(content="Weather in Paris?"),
        AssistantPromptMessage(content="", tool_calls=[call]),
        ToolPromptMessage(content='{"temp": 12}', tool_call_id="c1"),
    ]
    model = ScalewayLargeLanguageModel([CHAT_SCHEMA])
    counted = model.get_num_tokens(CHAT_MODEL, dict(PROVIDER_CREDS), messages)
    # Same estimate as with tool calling explicitly on, and more than the SDK
    # would count without it (it would drop the tool call).
    sdk = OAICompatLargeLanguageModel.get_num_tokens
    assert counted == sdk(model, CHAT_MODEL, dict(PROVIDER_CREDS, function_calling_type="tool_call"), messages)
    assert counted > sdk(model, CHAT_MODEL, dict(PROVIDER_CREDS), messages)


# ---------- embeddings ----------


def embedding_response(*args, **kwargs) -> MagicMock:
    n = len(json.loads(kwargs["data"])["input"])
    return ok_response(
        {
            "object": "list",
            "model": "m",
            "data": [{"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i in range(n)],
            "usage": {"prompt_tokens": n, "total_tokens": n},
        }
    )


@patch(EMB_POST, side_effect=embedding_response)
@patch.object(ScalewayTextEmbeddingModel, "_get_num_tokens_by_gpt2", gpt2_estimate)
def test_embeddings_batching_and_payload(mock_post):
    model = ScalewayTextEmbeddingModel([EMB_SCHEMA])
    result = model._invoke(EMB_MODEL, dict(PROVIDER_CREDS), [f"text {i}" for i in range(40)], user="u1")
    assert mock_post.call_count == 2  # 32 + 8, from max_chunks in the model schema
    first_call = mock_post.call_args_list[0]
    body = json.loads(first_call.kwargs["data"])
    assert first_call.args[0] == f"{BASE}/embeddings"
    assert len(body["input"]) == 32
    assert "user" not in body
    assert body["encoding_format"] == "float"  # sent by the SDK; Scaleway's default
    assert len(result.embeddings) == 40


def test_embeddings_added_model_uses_form_batch_size():
    model = ScalewayTextEmbeddingModel([])
    entity = model.get_customizable_model_schema(
        "new-embed", {"api_key": "k", "context_size": "8192", "max_chunks": "16"}
    )
    assert entity.model_properties[ModelPropertyKey.MAX_CHUNKS] == 16
    assert entity.model_properties[ModelPropertyKey.CONTEXT_SIZE] == 8192


# ---------- speech-to-text ----------


@patch(STT_POST, return_value=ok_response({"text": "hello world"}))
def test_speech_to_text_request(mock_post):
    model = ScalewaySpeech2TextModel([STT_SCHEMA])
    text = model._invoke(STT_MODEL, dict(PROVIDER_CREDS), io.BytesIO(b"fake-audio"), user="u1")
    assert mock_post.call_args.args[0] == f"{BASE}/audio/transcriptions"
    assert mock_post.call_args.kwargs["data"] == {"model": STT_MODEL}
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer scw-secret"
    assert text == "hello world"


def test_speech_to_text_added_model_schema():
    entity = ScalewaySpeech2TextModel([]).get_customizable_model_schema("new-stt", {"api_key": "k"})
    assert entity.model_properties[ModelPropertyKey.FILE_UPLOAD_LIMIT] == 25
    assert "mp3" in entity.model_properties[ModelPropertyKey.SUPPORTED_FILE_EXTENSIONS].split(",")
