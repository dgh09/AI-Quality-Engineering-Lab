"""Pruebas del cliente LLM compatible con OpenAI con el SDK simulado (sin red)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx2
import openai
import pytest
from openai.resources.chat import Completions

from rag_lab import llm as llm_module
from rag_lab.config import Settings
from rag_lab.llm import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    LLMClient,
    LLMError,
    OpenAICompatibleClient,
)

_API_KEY = "secret-key"

# Firmas reales del SDK: el doble las usa para que un parámetro mal escrito falle.
_OPENAI_INIT_SIGNATURE = inspect.signature(openai.OpenAI)
_CREATE_SIGNATURE = inspect.signature(Completions.create)


def _settings() -> Settings:
    return Settings(
        llm_base_url="http://llm.example:1234/v1",
        llm_api_key=_API_KEY,
        generator_model="gen-model",
        judge_model="judge-model",
        relevance_threshold=0.6,
        top_k=3,
        runs_per_case=3,
        pass_rate_threshold=0.67,
    )


def _completion(content: str | None) -> SimpleNamespace:
    """Respuesta mínima con la forma de `ChatCompletion` que usa el cliente."""
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    """Sustituto de `openai.OpenAI`: registra los kwargs del constructor y de `create`.

    Tanto el constructor como `chat.completions.create` validan sus kwargs contra las
    firmas reales del SDK (`bind` lanza `TypeError` ante un nombre desconocido). Las
    pruebas configuran y leen `self.create` (un `MagicMock`) como siempre.
    """

    instances: list[FakeOpenAI] = []

    def __init__(self, **kwargs: Any) -> None:
        _OPENAI_INIT_SIGNATURE.bind(**kwargs)
        self.init_kwargs = kwargs
        self.create = MagicMock(return_value=_completion("hello"))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._checked_create))
        FakeOpenAI.instances.append(self)

    def _checked_create(self, **kwargs: Any) -> Any:
        _CREATE_SIGNATURE.bind(None, **kwargs)  # None ocupa el lugar de `self`
        return self.create(**kwargs)


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> type[FakeOpenAI]:
    FakeOpenAI.instances = []
    monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAI)
    return FakeOpenAI


def test_client_is_built_with_base_url_api_key_timeout_and_no_retries_by_default(
    fake_sdk: type[FakeOpenAI],
) -> None:
    OpenAICompatibleClient(_settings(), model="gen-model")

    (sdk,) = fake_sdk.instances
    assert sdk.init_kwargs["base_url"] == "http://llm.example:1234/v1"
    assert sdk.init_kwargs["api_key"] == _API_KEY
    assert sdk.init_kwargs["timeout"] == DEFAULT_TIMEOUT_SECONDS
    assert sdk.init_kwargs["max_retries"] == DEFAULT_MAX_RETRIES == 0


def test_custom_timeout_and_max_retries_are_passed_to_the_sdk(
    fake_sdk: type[FakeOpenAI],
) -> None:
    OpenAICompatibleClient(_settings(), model="gen-model", timeout=5.0, max_retries=3)

    kwargs = fake_sdk.instances[0].init_kwargs
    assert kwargs["timeout"] == 5.0
    assert kwargs["max_retries"] == 3


@pytest.mark.parametrize("model", ["", "   "], ids=["empty", "blank"])
def test_empty_or_blank_model_is_rejected(fake_sdk: type[FakeOpenAI], model: str) -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleClient(_settings(), model=model)


def test_complete_sends_model_and_system_user_messages_and_returns_content(
    fake_sdk: type[FakeOpenAI],
) -> None:
    client = OpenAICompatibleClient(_settings(), model="judge-model")

    answer = client.complete("SYS", "USER")

    assert answer == "hello"
    kwargs = fake_sdk.instances[0].create.call_args.kwargs
    assert kwargs["model"] == "judge-model"
    assert kwargs["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert "response_format" not in kwargs


def test_json_mode_requests_json_object_response_format(fake_sdk: type[FakeOpenAI]) -> None:
    client = OpenAICompatibleClient(_settings(), model="judge-model")

    client.complete("SYS", "USER", json_mode=True)

    kwargs = fake_sdk.instances[0].create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


def test_fake_sdk_rejects_misspelled_parameters(fake_sdk: type[FakeOpenAI]) -> None:
    """Salvaguarda del propio doble: si no validara, las pruebas de parámetros no valdrían."""
    with pytest.raises(TypeError):
        FakeOpenAI(base_url="u", api_key="k", max_retry=0)
    sdk = FakeOpenAI(base_url="u", api_key="k")
    with pytest.raises(TypeError):
        sdk.chat.completions.create(model="m", messages=[], respons_format={})


def _not_found_error() -> openai.NotFoundError:
    response = httpx2.Response(404, request=httpx2.Request("POST", "http://x"))
    return openai.NotFoundError("model 'x' not found", response=response, body=None)


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=MagicMock()),
        openai.APITimeoutError(request=MagicMock()),
        _not_found_error(),
    ],
    ids=["connection", "timeout", "http-404"],
)
def test_sdk_errors_are_wrapped_in_llm_error_without_leaking_the_key(
    fake_sdk: type[FakeOpenAI], error: openai.OpenAIError
) -> None:
    client = OpenAICompatibleClient(_settings(), model="gen-model")
    fake_sdk.instances[0].create.side_effect = error

    with pytest.raises(LLMError) as excinfo:
        client.complete("SYS", "USER")

    assert excinfo.value.__cause__ is error
    assert "gen-model" in str(excinfo.value)
    assert _API_KEY not in str(excinfo.value)


@pytest.mark.parametrize("content", [None, "", "   "], ids=["none", "empty", "blank"])
def test_empty_content_raises_llm_error(
    fake_sdk: type[FakeOpenAI], content: str | None
) -> None:
    client = OpenAICompatibleClient(_settings(), model="gen-model")
    fake_sdk.instances[0].create.return_value = _completion(content)

    with pytest.raises(LLMError):
        client.complete("SYS", "USER")


def test_response_without_choices_raises_llm_error(fake_sdk: type[FakeOpenAI]) -> None:
    client = OpenAICompatibleClient(_settings(), model="gen-model")
    fake_sdk.instances[0].create.return_value = SimpleNamespace(choices=[])

    with pytest.raises(LLMError):
        client.complete("SYS", "USER")


def test_complete_signature_matches_the_llm_client_protocol() -> None:
    def shape(func: Any) -> list[tuple[str, inspect._ParameterKind, Any]]:
        return [
            (p.name, p.kind, p.default) for p in inspect.signature(func).parameters.values()
        ]

    expected = shape(LLMClient.complete)
    assert shape(OpenAICompatibleClient.complete) == expected
    assert ("json_mode", inspect.Parameter.POSITIONAL_OR_KEYWORD, False) in expected
