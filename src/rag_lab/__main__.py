"""CLI: `python -m rag_lab [--with-llm] [--out reports/report.md]`.

Indexa el corpus una vez en un Chroma en memoria, calcula las métricas deterministas
en ambos modos (shared = antes, isolated = después) y escribe el reporte Markdown.

- Sin `--with-llm` no se construye ningún cliente LLM: los agentes son solo de
  recuperación (`RagAgent(..., llm=None)`) y `deterministic_metrics` usa
  `RagAgent.retrieve`. Funciona con Ollama apagado.
- Con `--with-llm` se comprueba primero, ANTES de indexar nada, que el servidor
  responde y tiene cada modelo (`OpenAICompatibleClient.check_available`: lista de
  modelos, sin generar); si no, mensaje claro en stderr y código de salida 1 sin
  escribir nada. Después se ejecuta `run_suite` en ambos modos con
  `RUNS_PER_CASE` ejecuciones por caso. Los fallos del LLM *durante* la suite ya
  quedan registrados como ejecuciones fallidas (`RunError`); un `LLMError` que
  escapase igualmente se convierte en mensaje + código 1, no en un traceback.

Códigos de salida: 0 = reporte escrito; 1 = error de configuración, corpus o LLM;
2 = argumentos inválidos (argparse).
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

import chromadb

from rag_lab.agent import RagAgent
from rag_lab.config import Settings
from rag_lab.corpus import AGENT_IDS, CorpusError, GoldenCase, load_documents, load_golden_set
from rag_lab.evaluation import CaseResult, deterministic_metrics, run_suite
from rag_lab.judge import Judge
from rag_lab.llm import LLMClient, LLMError, OpenAICompatibleClient
from rag_lab.report import AFTER_LABEL, BEFORE_LABEL, ModeReport, format_rate, render_markdown
from rag_lab.store import VectorStore

DEFAULT_OUT = Path("reports") / "report.md"
EXIT_OK = 0
EXIT_ERROR = 1


class LLMUnavailableError(Exception):
    """El servidor LLM no respondió a la comprobación previa de `--with-llm`."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_lab",
        description=(
            "Evalúa los agentes RAG en modo shared (antes) e isolated (después) y "
            "escribe un reporte Markdown."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"ruta del reporte Markdown (por defecto: {DEFAULT_OUT.as_posix()})",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="añade la evaluación con generador y juez LLM (requiere Ollama en marcha)",
    )
    return parser


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_ERROR


def _build_store() -> VectorStore:
    """Indexa el corpus en una colección nueva de un Chroma en memoria (nada en disco).

    Nombre de colección único: dentro de un proceso los `EphemeralClient` comparten
    estado, así que dos ejecuciones de `main` no se pisan. No se borra ninguna
    colección (en chromadb 1.5.9 los borrados repetidos corrompen otras colecciones
    del mismo cliente; ver tests/conftest.py).
    """
    store = VectorStore(chromadb.EphemeralClient(), collection_name=f"cli-{uuid.uuid4().hex}")
    store.index(load_documents())
    return store


def _agents(
    store: VectorStore, settings: Settings, llm: LLMClient | None, isolated: bool
) -> dict[str, RagAgent]:
    return {
        agent_id: RagAgent(
            agent_id,
            store,
            llm,
            threshold=settings.relevance_threshold,
            top_k=settings.top_k,
            isolated=isolated,
        )
        for agent_id in AGENT_IDS
    }


def _connect_llms(settings: Settings) -> tuple[LLMClient, LLMClient]:
    """Crea y comprueba un cliente por modelo distinto → (generador, cliente del juez).

    Lanza `LLMUnavailableError` si el servidor no responde o falta algún modelo.
    """
    clients: dict[str, OpenAICompatibleClient] = {}
    for model in (settings.generator_model, settings.judge_model):
        if model in clients:
            continue
        client = OpenAICompatibleClient(settings, model)
        try:
            client.check_available()
        except LLMError as exc:
            raise LLMUnavailableError(
                f"el LLM no está disponible en {settings.llm_base_url} (modelo {model!r}). "
                f"¿Está Ollama en marcha y el modelo descargado? Detalle: {exc}"
            ) from exc
        clients[model] = client
    return clients[settings.generator_model], clients[settings.judge_model]


def _run_llm_suites(
    store: VectorStore,
    settings: Settings,
    cases: Sequence[GoldenCase],
    generator: LLMClient,
    judge_llm: LLMClient,
) -> tuple[tuple[CaseResult, ...], tuple[CaseResult, ...]]:
    """Evaluación con juez en ambos modos → (antes, después)."""
    judge = Judge(judge_llm)
    results = []
    for isolated in (False, True):
        results.append(
            run_suite(
                _agents(store, settings, generator, isolated),
                judge,
                cases,
                runs=settings.runs_per_case,
                pass_rate_threshold=settings.pass_rate_threshold,
            )
        )
    return results[0], results[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de la CLI; devuelve el código de salida (ver docstring del módulo)."""
    args = _build_parser().parse_args(argv)
    out: Path = args.out
    llms: tuple[LLMClient, LLMClient] | None = None
    try:
        settings = Settings.from_env()
        # Fallar rápido: comprobar el LLM antes de indexar (embeber) el corpus.
        if args.with_llm:
            llms = _connect_llms(settings)
        cases = load_golden_set()
        store = _build_store()
    except (ValueError, CorpusError, LLMUnavailableError) as exc:
        return _error(str(exc))

    before = deterministic_metrics(_agents(store, settings, None, isolated=False), cases)
    after = deterministic_metrics(_agents(store, settings, None, isolated=True), cases)

    before_llm: tuple[CaseResult, ...] | None = None
    after_llm: tuple[CaseResult, ...] | None = None
    if llms is not None:
        try:
            before_llm, after_llm = _run_llm_suites(store, settings, cases, *llms)
        except LLMError as exc:
            return _error(f"fallo del LLM durante la evaluación: {exc}")

    markdown = render_markdown(
        ModeReport(metrics=before, llm_results=before_llm),
        ModeReport(metrics=after, llm_results=after_llm),
    )
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        return _error(f"no se pudo escribir el reporte en {out}: {exc}")

    print(f"Reporte escrito en {out}")
    for label, metrics, llm_results in (
        (BEFORE_LABEL, before, before_llm),
        (AFTER_LABEL, after, after_llm),
    ):
        line = (
            f"  {label}: contaminación "
            f"{format_rate(len(metrics.contaminated_case_ids), metrics.total_cases)}"
        )
        if llm_results is not None:
            passed = sum(r.passed for r in llm_results)
            line += f" · casos que pasan (juez) {format_rate(passed, len(llm_results))}"
        print(line)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
