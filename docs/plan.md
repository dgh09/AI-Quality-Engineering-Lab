# Plan — AI Quality Engineering Lab (RAG con dos agentes)

Estado: **APROBADO (2026-09-22)** con cambios: segundo agente = `seguimiento`, corpus en inglés, modelo `qwen2.5:3b`, Ollama en Docker.

## 1. Resumen del diseño

- Dos agentes RAG sobre la misma empresa ficticia (**"Northwind Outfitters"**, tienda online):
  - `faq`: soporte al cliente (envíos, devoluciones, pagos, cuenta).
  - `seguimiento`: seguimiento de novedades de los empleados (vacaciones, horas extra, permisos, incapacidades, cambios de turno). Base de conocimiento interna de RR. HH.
- Un único cliente ChromaDB local. Cada chunk lleva `metadata.agent_id`.
- **Bug a propósito (modo `shared`)**: la consulta al vector store no filtra por `agent_id`, así que el agente FAQ recupera chunks internos de novedades de empleados (fuga de información interna a un canal de clientes).
- **Arreglo (modo `isolated`)**: la capa de datos (`store.query`) exige `agent_id` y aplica `where={"agent_id": ...}`. El prompt no participa en el aislamiento.
- **Abstención determinista**: el agente se abstiene *antes* de llamar al LLM si ningún chunk recuperado está por debajo de un umbral de distancia (`RELEVANCE_THRESHOLD`). Por eso las pruebas de abstención no necesitan LLM: en modo `shared` una pregunta de dominio cruzado encuentra chunks "relevantes" del otro agente y el agente responde (bug); en modo `isolated` no encuentra nada cercano y se abstiene (correcto).
- **LLM local** vía **Ollama corriendo en Docker** (contenedor `n8n-local-ollama-1`, puerto `11434`) usando su endpoint compatible con OpenAI (`openai` SDK). Modelo por defecto: `qwen2.5:3b`, configurable por variable de entorno para generador y juez por separado.
- **Juez LLM**: una llamada por ejecución devuelve JSON con tres criterios (`faithfulness`, `relevance`, `abstention`), cada uno `{score: 0-1, justification: str}`.
- **No determinismo**: cada caso se ejecuta 3 veces; un caso pasa si su tasa de éxito ≥ `PASS_RATE_THRESHOLD` (por defecto 0.67, es decir, 2 de 3).
- **Reporte** Markdown (`reports/report.md`) con métricas antes (`shared`) y después (`isolated`).

### Decisiones confirmadas
1. **Corpus y golden set en inglés** (el embedding local de Chroma, `all-MiniLM-L6-v2`, está entrenado en inglés). Código y documentación, en español.
2. **Ollama** corre en Docker (`n8n-local-ollama-1`, `http://localhost:11434`). Modelo `qwen2.5:3b` para generador y juez.
3. **API key**: el cliente lee `LLM_API_KEY` desde `.env` (relleno `ollama`); permite apuntar a otro proveedor compatible con OpenAI sin tocar código.
4. El modelo de embeddings (~80 MB) se descarga una vez la primera vez que se usa Chroma. No es una llamada a API.

## 2. Estructura de archivos

```
.
├── pyproject.toml            # dependencias, paquete src/, config de pytest (marker llm_eval, excluido por defecto)
├── .gitignore                # .env, .venv/, .chroma/, __pycache__/, reports/*.md generados salvo el del README
├── .env.example              # plantilla: LLM_BASE_URL, LLM_API_KEY, GENERATOR_MODEL, JUDGE_MODEL, umbrales
├── README.md                 # qué hace, cómo correrlo, resultados antes/después, Lecciones aprendidas
├── docs/plan.md              # este plan (se van marcando tareas)
├── data/
│   ├── faq_docs.json         # 8 documentos del agente faq   {id, agent_id, title, text}
│   ├── seguimiento_docs.json # 8 documentos del agente seguimiento
│   └── golden_set.json       # 12 casos {id, agent_id, question, category, expected_behavior, expected_source_ids}
├── scripts/
│   └── calibrate_threshold.py# imprime distancias top-1 por pregunta y modo para elegir el umbral
├── src/rag_lab/
│   ├── __init__.py
│   ├── config.py             # Settings (dataclass) desde entorno + .env; única fuente de configuración
│   ├── corpus.py             # carga y valida documentos y golden set (dataclasses Document, GoldenCase)
│   ├── store.py              # VectorStore sobre ChromaDB: index(), query(text, k, agent_id=None)
│   ├── contamination.py      # foreign_chunks(chunks, agent_id): detector puro de contaminación
│   ├── agent.py              # RagAgent: recuperar → compuerta de abstención → generar (LLM inyectado)
│   ├── llm.py                # LLMClient (Protocol) + OpenAICompatibleClient para Ollama; LLMError
│   ├── judge.py              # Judge: prompt de rúbrica, parseo/validación del JSON, reintento único
│   ├── evaluation.py         # run_case (3 ejecuciones, tasa de éxito), run_suite, métricas deterministas
│   ├── report.py             # render_markdown(antes, después) → str
│   └── __main__.py           # CLI: python -m rag_lab [--with-llm] [--out reports/report.md]
└── tests/
    ├── conftest.py           # fixtures: store efímero indexado, FakeLLM, golden set
    ├── test_smoke.py
    ├── test_config.py
    ├── test_corpus.py
    ├── test_store.py
    ├── test_agent.py
    ├── test_isolation.py     # EL núcleo: aserciones de agent_id y abstención en dominio cruzado, por modo
    ├── test_golden_deterministic.py  # golden set completo sin LLM: abstiene/responde según lo esperado
    ├── test_llm.py           # cliente con SDK simulado (sin red)
    ├── test_judge.py
    ├── test_evaluation.py
    ├── test_report.py
    ├── test_cli.py
    └── llm_eval/
        └── test_quality.py   # @pytest.mark.llm_eval: juez real sobre golden set, 3 ejecuciones por caso
```

## 3. Convenciones del proyecto (se pasan a cada subagente)

- Python 3.11+ (la máquina tiene 3.13), layout `src/`, venv en `.venv`, instalación `pip install -e ".[dev]"`.
- Shell: Windows PowerShell. Comandos de verificación: `.venv\Scripts\python -m pytest ...`.
- Type hints en todo; `@dataclass(frozen=True)` para objetos de valor; sin clases donde basta una función.
- Nada de red en pruebas por defecto: ChromaDB con `EphemeralClient`, LLM con `FakeLLM` de `conftest.py`.
- Pruebas que llaman al LLM real: solo en `tests/llm_eval/`, con `@pytest.mark.llm_eval`. `pytest` a secas las excluye (`addopts = -m "not llm_eval"`).
- Configuración solo a través de `config.Settings`; nunca claves ni URLs de proveedor en código fuera de defaults locales.
- Errores: excepciones propias y explícitas (`LLMError`, `JudgeError`, `CorpusError`); nada de `except Exception: pass`.
- TDD estricto: prueba primero, verla fallar (pegar la salida), implementar, verla pasar. Prohibido editar pruebas existentes para que pasen.
- Nombres de pruebas describen comportamiento: `test_faq_agent_abstains_on_seguimiento_question_when_isolated`.
- Commits: `tipo(ámbito): descripción` (p. ej. `feat(store): add agent_id filter to query`).

## 4. Tareas

Formato: **ID — título** · depende de · verificación.

- [x] **T0 — Bootstrap del repo** · depende de: — 
  `git init`, `pyproject.toml` (chromadb, openai, python-dotenv; dev: pytest), `.gitignore`, `.env.example`, paquete vacío `rag_lab`, `tests/test_smoke.py` (importa `rag_lab`), marker `llm_eval` registrado y excluido por defecto.
  *Verificación*: `pytest` → `1 passed`; `git check-ignore .env` imprime `.env`; `pytest --markers` lista `llm_eval`.

- [x] **T1 — Configuración** · depende de: T0
  `config.py` con `Settings.from_env()`: `LLM_BASE_URL` (def. `http://localhost:11434/v1`), `LLM_API_KEY` (def. `ollama`), `GENERATOR_MODEL` y `JUDGE_MODEL` (def. `qwen2.5:3b`), `RELEVANCE_THRESHOLD`, `TOP_K` (def. 3), `RUNS_PER_CASE` (def. 3), `PASS_RATE_THRESHOLD` (def. 0.67). Carga `.env` con python-dotenv. Valores inválidos → `ValueError` claro.
  *Verificación*: `pytest tests/test_config.py` — defaults, override por `monkeypatch.setenv`, valor inválido lanza error.

- [x] **T2 — Corpus de documentos** · depende de: T0
  `data/faq_docs.json` y `data/seguimiento_docs.json` (8 docs cortos cada uno, en inglés), `corpus.load_documents()` que valida ids únicos, `agent_id` ∈ {faq, seguimiento}, texto no vacío → `CorpusError`.
  *Verificación*: `pytest tests/test_corpus.py` — 8+8 docs, ids únicos, `agent_id` correcto por archivo, JSON inválido lanza `CorpusError`.

- [x] **T3 — Golden set** · depende de: T2
  `data/golden_set.json` con 12 casos: 4 in-domain faq, 4 in-domain seguimiento, 2 de dominio cruzado (uno por agente, `expected_behavior: abstain`), 2 fuera de dominio (`abstain`). `corpus.load_golden_set()` valida que los `expected_source_ids` existan y pertenezcan al mismo `agent_id`.
  *Verificación*: `pytest tests/test_corpus.py` — 12 casos, distribución por categoría, referencias válidas.

- [x] **T4 — Vector store (versión con el bug)** · depende de: T2
  `store.VectorStore(client)`: `index(documents)` en **una sola colección** con `metadata.agent_id` y distancia coseno; `query(text, k)` → `list[Chunk(id, agent_id, text, distance)]` ordenada por distancia. Sin filtro por agente (bug intencional, documentado en docstring).
  *Verificación*: `pytest tests/test_store.py` — indexa 16, `query` devuelve `k` chunks ordenados con metadata; una pregunta de envíos trae un doc faq en top-1.

- [x] **T5 — Agente RAG con compuerta de abstención** · depende de: T4, T1
  `agent.RagAgent(agent_id, store, llm, threshold, top_k, isolated: bool)` → `AgentResponse(answer, abstained, retrieved, context)` (`retrieved` = todo el top-k; `context` = los chunks con `distance <= threshold` que ve el LLM). Si ningún chunk tiene `distance <= threshold`, se abstiene con un mensaje fijo **sin llamar al LLM**. Si no, construye el prompt con los chunks relevantes y llama a `llm.complete`. En esta tarea `isolated` todavía no filtra (`isolated=True` lanza `NotImplementedError` al construir el agente). El prompt pide responder exactamente `ABSTENTION_MESSAGE` si el contexto no tiene la respuesta. `FakeLLM` vive en `tests/fakes.py`.
  *Verificación*: `pytest tests/test_agent.py` — con `FakeLLM`: abstiene y no llama al LLM cuando todo está lejos; responde y pasa el contexto al LLM cuando hay chunks cercanos.

- [x] **T6 — Detector y pruebas que reproducen la contaminación** · depende de: T3, T5
  `contamination.foreign_chunks(chunks, agent_id)`. `tests/test_isolation.py` con las aserciones **correctas** parametrizadas por modo; en esta tarea solo existe el modo `shared`, marcado `xfail(strict=True, reason="BUG: vector store compartido sin aislamiento")`:
  (a) ningún chunk recuperado por el agente faq tiene `agent_id != "faq"` para las preguntas del golden set del agente faq;
  (b) el agente faq se abstiene en la pregunta de dominio cruzado.
  Además, una prueba no-xfail que afirma explícitamente que en `shared` la pregunta cruzada trae chunks `seguimiento` (reproducción del bug).
  *Verificación*: `pytest tests/test_isolation.py -rxX` → reproducción `passed`, aserciones correctas `xfailed` (fallan de verdad, y strict impide que pasen en silencio).

- [x] **T7 — Arreglo: aislamiento en la capa de datos** · depende de: T6
  `store.query(text, k, *, agent_id)` (obligatorio, solo por nombre; `None` = modo shared) aplica `where={"agent_id": agent_id}` cuando no es `None`; `RagAgent(isolated=True)` siempre lo pasa. Se **añade** el parámetro `isolated` a `test_isolation.py` (las aserciones existentes no se tocan).
  *Verificación*: `pytest tests/test_isolation.py tests/test_store.py -rxX` → casos `isolated` `passed`, casos `shared` siguen `xfailed`.

- [ ] **T8 — Calibración del umbral y golden set determinista** · depende de: T7
  `scripts/calibrate_threshold.py` imprime la distancia top-1 de cada pregunta en ambos modos. Se fija el default de `RELEVANCE_THRESHOLD` en `config.py` con la justificación en un comentario. `tests/test_golden_deterministic.py`: en modo `isolated`, con `FakeLLM`, los 12 casos se abstienen o responden según `expected_behavior`, y los in-domain recuperan su `expected_source_ids` en top-k.
  *Verificación*: salida del script pegada en el reporte del subagente + `pytest tests/test_golden_deterministic.py` → 12 passed.
  *Condición de parada*: si ningún umbral separa limpiamente los casos, el implementador se detiene y reporta las distancias (no ajusta datos por su cuenta).

- [ ] **T9 — Cliente LLM** · depende de: T1 *(independiente de T2–T8)*
  `llm.LLMClient` (Protocol `complete(system, user, json_mode=False) -> str`) y `OpenAICompatibleClient(settings, model)` usando el SDK `openai` con `base_url` y `api_key` de `Settings`; `json_mode` → `response_format={"type": "json_object"}`; timeout; errores del SDK → `LLMError`.
  *Verificación*: `pytest tests/test_llm.py` — SDK simulado: parámetros correctos (modelo, base_url, response_format), error de conexión → `LLMError`. Sin red.

- [ ] **T10 — Juez LLM con rúbrica** · depende de: T9
  `judge.Judge(llm).evaluate(question, context, answer, expected_behavior) -> Verdict` con `faithfulness`, `relevance`, `abstention` (`Score(score: float, justification: str)`). Prompt con rúbrica explícita y formato JSON exigido. JSON inválido o score fuera de [0,1] → un reintento, luego `JudgeError`. `Verdict.passed(expected_behavior, min_score=0.7)`: `answer` → fidelidad y relevancia ≥ min; `abstain` → abstención ≥ min.
  *Verificación*: `pytest tests/test_judge.py` — JSON válido parseado, inválido→reintento→éxito, inválido dos veces→`JudgeError`, score 1.4→rechazado, reglas de `passed`.

- [ ] **T11 — Evaluación con repeticiones** · depende de: T5, T10
  `evaluation.run_case(agent, judge, case, runs) -> CaseResult(pass_rate, passed, verdicts)` y `run_suite(...)`; un error del juez cuenta como ejecución fallida (se registra, no se oculta). `deterministic_metrics(agent_by_id, cases)` → tasa de contaminación y exactitud de abstención (sin LLM).
  *Verificación*: `pytest tests/test_evaluation.py` — con fakes secuenciales: 2/3 → pasa con umbral 0.67, 1/3 → falla; `JudgeError` cuenta como fallo; métricas deterministas correctas sobre un escenario fijo.

- [ ] **T12 — Reporte Markdown** · depende de: T11
  `report.render_markdown(before, after) -> str`: tabla de métricas deterministas (contaminación, abstención) antes/después y, si existen, tabla de evaluación LLM por caso (tasa de éxito, pasa/no pasa, justificación breve del juez).
  *Verificación*: `pytest tests/test_report.py` — contiene secciones "Antes (shared)" y "Después (isolated)", números esperados, y omite la sección LLM si no hay resultados.

- [ ] **T13 — CLI** · depende de: T12
  `python -m rag_lab --out reports/report.md` corre métricas deterministas en ambos modos y escribe el reporte; `--with-llm` añade la evaluación con juez. Si Ollama no responde, mensaje claro y código de salida ≠ 0.
  *Verificación*: `pytest tests/test_cli.py` (modo determinista sobre `tmp_path`) y `.venv\Scripts\python -m rag_lab --out reports/report.md` crea el archivo con contaminación > 0 antes y = 0 después.

- [ ] **T14 — Evaluaciones con LLM real** · depende de: T13
  `tests/llm_eval/test_quality.py` con `@pytest.mark.llm_eval`: corre la suite en modo `isolated` con el juez real, 3 ejecuciones por caso, y afirma que la proporción de casos que pasan ≥ `PASS_RATE_THRESHOLD`. `skip` con motivo claro si Ollama no está disponible.
  *Verificación*: `pytest` → no las ejecuta (deselected); `pytest -m llm_eval` → skip limpio sin Ollama, o resultado real con Ollama.

- [ ] **T15 — README** · depende de: T13 (y T14 si hay Ollama)
  Qué hace, requisitos (incl. Ollama en Docker y `docker exec <contenedor> ollama pull qwen2.5:3b`), instalación, cómo correr pruebas deterministas / `llm_eval` / reporte, resultados antes/después (pegados del reporte generado) y sección vacía **"Lecciones aprendidas"**.
  *Verificación*: seguir los comandos del README desde cero en un venv limpio funciona; el revisor final lo confirma.

- [ ] **REVISIÓN FINAL** · depende de: todo
  Coherencia entre módulos, `pytest` completo en verde, README ejecutable.

## 5. Dependencias y paralelismo

```
T0 ─┬─ T1 ─┬──────────────── T9 ── T10 ─┐
    │      │                             │
    └─ T2 ─┼─ T3 ─────────┐              │
           └─ T4 ── T5 ───┴─ T6 ── T7 ── T8
                   │                     
                   └─────────────────────┴─ T11 ── T12 ── T13 ── T14 ── T15
```

- Independientes entre sí: **T1 ∥ T2**, **T3 ∥ T4**, y la rama **T9 → T10** es independiente de toda la rama de retrieval (T2–T8).
- Ruta crítica: T0 → T2 → T4 → T5 → T6 → T7 → T8 → T11 → T12 → T13 → T15.
- Aun así ejecutaré las tareas en el orden numérico, una por una (commit por tarea), para mantener el historial y las revisiones simples.

## 6. Riesgos

- **Separación de distancias con MiniLM**: si el umbral no separa in-domain de cruzado/fuera de dominio, T8 se detiene y lo consultamos (probable ajuste de redacción de documentos o preguntas).
- **Juez pequeño poco fiable**: un modelo de 3B puede puntuar de forma ruidosa; por eso se mide tasa de éxito sobre 3 ejecuciones y el modelo es configurable.
- **Tiempo de `llm_eval`**: 12 casos × 3 ejecuciones × 2 llamadas (generador + juez) ≈ 72 llamadas locales por modo.
