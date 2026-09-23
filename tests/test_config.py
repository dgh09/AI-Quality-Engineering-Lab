from pathlib import Path

import pytest

from rag_lab.config import Settings

ENV_VARS: tuple[str, ...] = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "GENERATOR_MODEL",
    "JUDGE_MODEL",
    "RELEVANCE_THRESHOLD",
    "TOP_K",
    "RUNS_PER_CASE",
    "PASS_RATE_THRESHOLD",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_when_env_is_empty(clean_env: pytest.MonkeyPatch) -> None:
    settings = Settings.from_env(env_file=None)

    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_api_key == "ollama"
    assert settings.generator_model == "qwen2.5:3b"
    assert settings.judge_model == "qwen2.5:3b"
    assert settings.relevance_threshold == 0.6
    assert settings.top_k == 3
    assert settings.runs_per_case == 3
    assert settings.pass_rate_threshold == 0.67


def test_settings_is_frozen(clean_env: pytest.MonkeyPatch) -> None:
    settings = Settings.from_env(env_file=None)

    with pytest.raises(AttributeError):
        settings.top_k = 10  # type: ignore[misc]


def test_overrides_from_environment(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_BASE_URL", "http://example.test:8000/v1")
    clean_env.setenv("LLM_API_KEY", "test-key")
    clean_env.setenv("GENERATOR_MODEL", "gen-model")
    clean_env.setenv("JUDGE_MODEL", "judge-model")
    clean_env.setenv("RELEVANCE_THRESHOLD", "0.8")
    clean_env.setenv("TOP_K", "5")
    clean_env.setenv("RUNS_PER_CASE", "7")
    clean_env.setenv("PASS_RATE_THRESHOLD", "1")

    settings = Settings.from_env(env_file=None)

    assert settings.llm_base_url == "http://example.test:8000/v1"
    assert settings.llm_api_key == "test-key"
    assert settings.generator_model == "gen-model"
    assert settings.judge_model == "judge-model"
    assert settings.relevance_threshold == 0.8
    assert settings.top_k == 5
    assert settings.runs_per_case == 7
    assert settings.pass_rate_threshold == 1.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RELEVANCE_THRESHOLD", "abc"),
        ("TOP_K", "tres"),
        ("TOP_K", "2.5"),
        ("RUNS_PER_CASE", ""),
        ("PASS_RATE_THRESHOLD", "alto"),
    ],
)
def test_non_numeric_value_raises(
    clean_env: pytest.MonkeyPatch, name: str, value: str
) -> None:
    clean_env.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        Settings.from_env(env_file=None)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RELEVANCE_THRESHOLD", "0"),
        ("RELEVANCE_THRESHOLD", "-0.1"),
        ("TOP_K", "0"),
        ("RUNS_PER_CASE", "-1"),
        ("PASS_RATE_THRESHOLD", "1.01"),
        ("PASS_RATE_THRESHOLD", "-0.01"),
    ],
)
def test_out_of_range_value_raises(
    clean_env: pytest.MonkeyPatch, name: str, value: str
) -> None:
    clean_env.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        Settings.from_env(env_file=None)


def test_env_file_values_are_loaded_without_overriding_environment(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GENERATOR_MODEL=from-file-gen\n"
        "JUDGE_MODEL=from-file-judge\n"
        "TOP_K=4\n",
        encoding="utf-8",
    )
    clean_env.setenv("JUDGE_MODEL", "from-env-judge")

    settings = Settings.from_env(env_file=env_file)

    assert settings.generator_model == "from-file-gen"
    assert settings.top_k == 4
    assert settings.judge_model == "from-env-judge"


def test_missing_env_file_falls_back_to_defaults(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings.from_env(env_file=tmp_path / "does-not-exist.env")

    assert settings.top_k == 3
    assert settings.generator_model == "qwen2.5:3b"


def test_api_key_is_not_exposed_in_repr_or_str(
    clean_env: pytest.MonkeyPatch,
) -> None:
    secret = "sk-super-secret-value-123"
    clean_env.setenv("LLM_API_KEY", secret)

    settings = Settings.from_env(env_file=None)

    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert settings.llm_api_key == secret


def test_from_env_does_not_mutate_os_environ(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    env_file = tmp_path / ".env"
    env_file.write_text("GENERATOR_MODEL=only-in-file\n", encoding="utf-8")

    settings = Settings.from_env(env_file=env_file)

    assert settings.generator_model == "only-in-file"
    assert "GENERATOR_MODEL" not in os.environ


@pytest.mark.parametrize("name", ["RELEVANCE_THRESHOLD", "PASS_RATE_THRESHOLD"])
@pytest.mark.parametrize("value", ["inf", "-inf", "nan"])
def test_non_finite_float_raises(
    clean_env: pytest.MonkeyPatch, name: str, value: str
) -> None:
    clean_env.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        Settings.from_env(env_file=None)
