"""Pruebas de la CLI (`python -m rag_lab`), llamando a `main(argv)` sin subprocess ni red."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rag_lab.__main__ as cli
import rag_lab.llm as llm_module
from rag_lab.agent import RagAgent
from rag_lab.llm import LLMError
from rag_lab.report import AFTER_LABEL, BEFORE_LABEL
from rag_lab.store import Chunk, VectorStore

_SETTINGS_VARS = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "GENERATOR_MODEL",
    "JUDGE_MODEL",
    "RELEVANCE_THRESHOLD",
    "TOP_K",
    "RUNS_PER_CASE",
    "PASS_RATE_THRESHOLD",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Defaults de Settings: sin variables del entorno y sin el .env del repo (CWD = tmp)."""
    for name in _SETTINGS_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def forbid_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hace fallar la prueba si algo construye un cliente LLM (CLI o SDK `openai`)."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("el modo determinista no debe construir ningún cliente LLM")

    monkeypatch.setattr(cli, "OpenAICompatibleClient", _boom)
    monkeypatch.setattr(llm_module, "OpenAICompatibleClient", _boom)
    monkeypatch.setattr(llm_module, "OpenAI", _boom)


class _MemoizedStore:
    """Envoltorio de un store de SOLO LECTURA que memoriza `query` (los `Chunk` son
    inmutables). Lo caro de la CLI con el store ya indexado es embeber cada pregunta;
    las mismas 12 preguntas se repiten en cada modo y en cada prueba."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._cache: dict[tuple[str, int, str | None], list[Chunk]] = {}

    def query(self, text: str, k: int, *, agent_id: str | None) -> list[Chunk]:
        key = (text, k, agent_id)
        if key not in self._cache:
            self._cache[key] = self._store.query(text, k, agent_id=agent_id)
        return list(self._cache[key])


@pytest.fixture(scope="module")
def memoized_store(indexed_store: VectorStore) -> _MemoizedStore:
    return _MemoizedStore(indexed_store)


@pytest.fixture
def fast_store(monkeypatch: pytest.MonkeyPatch, memoized_store: _MemoizedStore) -> None:
    """Sustituye `_build_store` por el store de sesión ya indexado (solo lectura), con
    las consultas memorizadas.

    Evita re-embeber el corpus y las preguntas en cada prueba; las pruebas end-to-end
    de contaminación e idempotencia siguen usando el `_build_store` real.
    """
    monkeypatch.setattr(cli, "_build_store", lambda: memoized_store)


@pytest.fixture
def forbid_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hace fallar la prueba si se construye el store (se embebería el corpus)."""

    def _boom() -> None:
        raise AssertionError("no se debe construir el store antes de comprobar el LLM")

    monkeypatch.setattr(cli, "_build_store", _boom)


def _metrics_row(report: str, metric: str) -> list[str]:
    line = next(line for line in report.splitlines() if line.startswith(f"| {metric} |"))
    return [cell.strip() for cell in line.strip("|").split("|")]


# --- Modo determinista --------------------------------------------------------------


def test_deterministic_mode_writes_report_with_contamination_only_before(
    tmp_path: Path, forbid_llm: None, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "nested" / "dir" / "report.md"

    code = cli.main(["--out", str(out)])

    assert code == 0
    report = out.read_text(encoding="utf-8")
    assert BEFORE_LABEL in report and AFTER_LABEL in report
    _, before, after = _metrics_row(report, "Tasa de contaminación")
    assert not before.startswith("0.0%"), f"shared debería estar contaminado: {before}"
    assert after.startswith("0.0% ("), f"isolated no debería estar contaminado: {after}"
    assert "Evaluación con juez LLM" not in report
    stdout = capsys.readouterr().out
    assert str(out) in stdout
    assert before in stdout and after in stdout


def test_deterministic_mode_accepts_relative_out_path(
    tmp_path: Path, forbid_llm: None, fast_store: None
) -> None:
    code = cli.main(["--out", "reports/report.md"])

    assert code == 0
    assert (tmp_path / "reports" / "report.md").is_file()


def test_deterministic_mode_is_idempotent_within_one_process(
    tmp_path: Path, forbid_llm: None
) -> None:
    first, second = tmp_path / "a.md", tmp_path / "b.md"

    assert cli.main(["--out", str(first)]) == 0
    assert cli.main(["--out", str(second)]) == 0

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_retrieval_only_agent_refuses_to_generate(indexed_store) -> None:  # type: ignore[no-untyped-def]
    agent = RagAgent("faq", indexed_store, None, threshold=2.0, top_k=3, isolated=True)

    assert agent.retrieve("How much does express shipping cost?").context
    with pytest.raises(RuntimeError, match="sin cliente LLM"):
        agent.ask("How much does express shipping cost?")


# --- Argumentos ---------------------------------------------------------------------


def test_unknown_argument_exits_with_argparse_code_2() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--nope"])
    assert excinfo.value.code == 2


def test_out_without_value_exits_with_argparse_code_2() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--out"])
    assert excinfo.value.code == 2


def test_invalid_settings_exit_non_zero_with_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TOP_K", "cero")
    monkeypatch.setattr(cli, "_build_store", lambda: pytest.fail("no debe indexar"))
    out = tmp_path / "report.md"

    code = cli.main(["--out", str(out)])

    assert code != 0
    assert "TOP_K" in capsys.readouterr().err
    assert not out.exists()


def test_out_pointing_at_existing_directory_exits_1_with_message(
    tmp_path: Path, forbid_llm: None, fast_store: None, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "ya-soy-un-directorio"
    target.mkdir()

    code = cli.main(["--out", str(target)])

    assert code == 1
    err = capsys.readouterr().err
    assert "no se pudo escribir el reporte" in err
    assert "Traceback" not in err


# --- Modo --with-llm ----------------------------------------------------------------


class _DownClient:
    """Cliente que simula Ollama caído: la comprobación y toda llamada lanzan `LLMError`."""

    def __init__(self, settings: object, model: str, **kwargs: object) -> None:
        self.model = model

    def check_available(self) -> None:
        raise LLMError(f"No se pudo listar los modelos para {self.model!r}: APIConnectionError")

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        raise AssertionError("no se debe llamar a complete si la comprobación falla")


_VALID_VERDICT = json.dumps(
    {
        dim: {"score": 1.0, "justification": f"{dim} ok"}
        for dim in ("faithfulness", "relevance", "abstention")
    }
)


class _UpClient:
    """Cliente que simula Ollama disponible: JSON válido para el juez, texto para el generador.

    `checked` registra el modelo de cada `check_available`.
    """

    checked: list[str] = []

    def __init__(self, settings: object, model: str, **kwargs: object) -> None:
        self.model = model

    def check_available(self) -> None:
        _UpClient.checked.append(self.model)

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return _VALID_VERDICT if json_mode else "OK"


@pytest.fixture
def up_client(monkeypatch: pytest.MonkeyPatch) -> type[_UpClient]:
    _UpClient.checked = []
    monkeypatch.setattr(cli, "OpenAICompatibleClient", _UpClient)
    return _UpClient


def test_with_llm_exits_non_zero_with_clear_message_when_ollama_is_down(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    forbid_store: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleClient", _DownClient)
    monkeypatch.setattr(llm_module, "OpenAI", None)  # ni siquiera el SDK real
    out = tmp_path / "report.md"

    code = cli.main(["--with-llm", "--out", str(out)])

    assert code == 1
    err = capsys.readouterr().err
    assert "Ollama" in err
    assert "http://localhost:11434/v1" in err
    assert "Traceback" not in err
    assert not out.exists()


@pytest.mark.parametrize("name", ["GENERATOR_MODEL", "JUDGE_MODEL"])
def test_with_llm_and_empty_model_exits_1_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    forbid_llm: None,
    forbid_store: None,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    monkeypatch.setenv(name, "")
    out = tmp_path / "report.md"

    code = cli.main(["--with-llm", "--out", str(out)])

    assert code == 1
    err = capsys.readouterr().err
    assert name in err
    assert "Traceback" not in err
    assert not out.exists()


def test_with_llm_turns_unexpected_llm_error_into_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    up_client: type[_UpClient],
    fast_store: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _fail(*args: object, **kwargs: object) -> None:
        raise LLMError("se cortó la conexión a mitad de la suite")

    monkeypatch.setattr(cli, "run_suite", _fail)
    out = tmp_path / "report.md"

    code = cli.main(["--with-llm", "--out", str(out)])

    assert code == 1
    assert "se cortó la conexión" in capsys.readouterr().err
    assert not out.exists()


def test_with_llm_adds_judge_section_for_both_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, up_client: type[_UpClient], fast_store: None
) -> None:
    monkeypatch.setenv("RUNS_PER_CASE", "1")
    out = tmp_path / "report.md"

    code = cli.main(["--with-llm", "--out", str(out)])

    assert code == 0
    # Generador y juez usan el mismo modelo por defecto: una sola comprobación.
    assert up_client.checked == ["qwen2.5:3b"]
    report = out.read_text(encoding="utf-8")
    assert "Evaluación con juez LLM" in report
    assert f"### {BEFORE_LABEL}" in report and f"### {AFTER_LABEL}" in report
    assert "_Sin resultados de evaluación LLM" not in report
    # 12 casos con 1 ejecución por modo: todas las filas presentes en ambos modos.
    assert report.count("| gs-01 |") == 2


def test_with_llm_checks_each_distinct_model_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, up_client: type[_UpClient], fast_store: None
) -> None:
    monkeypatch.setenv("RUNS_PER_CASE", "1")
    monkeypatch.setenv("GENERATOR_MODEL", "gen-model")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")

    code = cli.main(["--with-llm", "--out", str(tmp_path / "report.md")])

    assert code == 0
    assert up_client.checked == ["gen-model", "judge-model"]
