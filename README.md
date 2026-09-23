# AI Quality Engineering Lab — RAG con dos agentes

Laboratorio de ingeniería de calidad para un sistema RAG multi-agente. Muestra de principio a fin cómo **introducir un bug a propósito, detectarlo con pruebas, arreglarlo en la capa de datos y medir el antes y el después**, con pruebas deterministas (sin LLM) y una evaluación con un juez LLM local.

## Qué hace

- Dos agentes RAG sobre la misma empresa ficticia, **Northwind Outfitters** (tienda online):
  - `faq`: soporte al cliente (envíos, devoluciones, pagos, cuenta).
  - `seguimiento`: base interna de RR. HH. sobre novedades de empleados (vacaciones, horas extra, permisos, incapacidades, cambios de turno).
- Un único vector store **ChromaDB** con los 16 documentos (8 por agente, en inglés); cada chunk lleva `metadata.agent_id`.
- **Compuerta de abstención determinista**: si ningún chunk recuperado está a una distancia coseno ≤ `RELEVANCE_THRESHOLD`, el agente se abstiene con un mensaje fijo **sin llamar al LLM**.
- **Golden set** de 12 casos (`data/golden_set.json`): 4 in-domain por agente, 2 de dominio cruzado y 2 fuera de dominio (estos 4 deben abstenerse).
- **LLM local** (Ollama, endpoint compatible con OpenAI) como generador y como **juez** con rúbrica (`faithfulness`, `relevance`, `abstention`, cada una 0–1 con justificación, salida JSON validada).
- **Reporte Markdown** con las métricas antes (modo `shared`) y después (modo `isolated`).

### El bug intencional, cómo se detecta y cómo se arregla

- **Bug (modo `shared`)**: el vector store es compartido y la consulta **no filtra por `agent_id`**, así que la contaminación de contexto va en ambos sentidos: cada agente puede recuperar chunks del otro. El caso grave es `faq`, el canal de clientes, recuperando chunks internos de RR. HH. del agente `seguimiento` (fuga de información interna a un canal externo). Además, las dos preguntas de dominio cruzado (gs-09, gs-10) encuentran chunks "relevantes" del otro agente, pasan la compuerta y responden en lugar de abstenerse.
- **Detección**:
  - `tests/test_isolation.py` contiene las aserciones del comportamiento **correcto** (ningún chunk ajeno; abstención en la pregunta cruzada). En modo `shared` están marcadas `xfail(strict=True)`: fallan de verdad, y si alguien "arreglara" el bug sin querer la suite lo señalaría (XPASS → fallo). Una prueba aparte reproduce el bug explícitamente.
  - Una **métrica determinista de contaminación** sobre la recuperación (`contamination.foreign_chunks`, `evaluation.deterministic_metrics`): un caso está contaminado si su top-k contiene algún chunk de otro agente.
- **Arreglo (modo `isolated`)**: el aislamiento vive en la **capa de datos**, no en el prompt. `VectorStore.query(text, k, *, agent_id)` aplica `where={"agent_id": ...}` y `RagAgent(isolated=True)` siempre pasa su `agent_id`.
- **Ojo con el juez**: en modo `shared` una puntuación alta de *faithfulness* **no implica ausencia de contaminación**. El juez puntúa la respuesta contra el contexto que vio el generador; si ese contexto ya estaba contaminado, una respuesta copiada de un chunk ajeno es perfectamente "fiel". Por eso la contaminación se mide sobre la recuperación, nunca con el juez.

La historia completa está en `git log --oneline` (commits pequeños por tarea, Tn) y el plan en [`docs/plan.md`](docs/plan.md).

## Requisitos

- **Python 3.11+** (probado con 3.13 en Windows 11).
- **Red en la primera ejecución**: ChromaDB usa su `DefaultEmbeddingFunction` (`all-MiniLM-L6-v2` en ONNX, ~80 MB), que se descarga la primera vez que se indexa el corpus y queda en caché. No es una llamada a una API.
- **Solo para la evaluación con LLM** (opcional; las pruebas deterministas y el reporte determinista no lo necesitan):
  - **Docker** con **Ollama** escuchando en `http://localhost:11434`. Si aún no tienes un contenedor:

    ```powershell
    docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
    ```

  - El modelo `qwen2.5:3b` descargado dentro del contenedor:

    ```powershell
    docker exec ollama ollama pull qwen2.5:3b
    ```

    Si tu contenedor tiene otro nombre (por ejemplo `n8n-local-ollama-1`), sustitúyelo.

## Instalación

Desde la raíz del repositorio, en PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

En Linux/macOS: `python3 -m venv .venv`, `.venv/bin/python -m pip install -e ".[dev]"` y `cp .env.example .env`.

> **Windows y rutas largas.** `onnxruntime` (dependencia de ChromaDB) instala archivos con rutas muy largas. Si el repositorio está en una carpeta profunda y Windows no tiene habilitadas las rutas largas, `pip` falla con `OSError: [Errno 2] No such file or directory ... onnxruntime\tools\...`. Solución: clonar en una ruta corta (p. ej. `C:\dev\...`) o habilitar `LongPathsEnabled` ([guía de pip](https://pip.pypa.io/warnings/enable-long-paths)).

### Configuración (`.env`)

Toda la configuración pasa por `rag_lab.config.Settings`, que lee variables de entorno y, si existe, el archivo `.env` (el entorno real tiene prioridad). El `.env` es opcional: sin él se usan los valores por defecto. Nunca se commitea (está en `.gitignore`).

| Variable | Por defecto | Para qué sirve |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Endpoint compatible con OpenAI (Ollama). |
| `LLM_API_KEY` | `ollama` | Ollama la ignora; permite apuntar a otro proveedor compatible sin tocar código. |
| `GENERATOR_MODEL` | `qwen2.5:3b` | Modelo que genera las respuestas. |
| `JUDGE_MODEL` | `qwen2.5:3b` | Modelo del juez. |
| `RELEVANCE_THRESHOLD` | `0.6` | Distancia coseno máxima para que un chunk entre al contexto (calibrado con `scripts/calibrate_threshold.py`). |
| `TOP_K` | `3` | Chunks recuperados por pregunta. |
| `RUNS_PER_CASE` | `3` | Ejecuciones por caso en la evaluación con LLM (generador y juez no son deterministas). |
| `PASS_RATE_THRESHOLD` | `0.67` | Tasa de éxito mínima para que un caso pase (2 de 3), y proporción mínima de casos que pasan en `llm_eval`. |

## Cómo correrlo

### Pruebas deterministas (sin LLM, sin red salvo la descarga inicial del embedding)

```powershell
.venv\Scripts\python -m pytest
```

Resultado esperado: `335 passed, 1 deselected, 2 xfailed`. Los 2 `xfailed` son las aserciones correctas en modo `shared` (el bug documentado); el `deselected` es la evaluación con LLM, excluida por defecto (`addopts = -m 'not llm_eval'`). Para ver el detalle de los xfail:

```powershell
.venv\Scripts\python -m pytest tests/test_isolation.py -rxX
```

### Evaluación con LLM real (`llm_eval`)

```powershell
.venv\Scripts\python -m pytest -m llm_eval -s
```

- Corre el golden set en modo `isolated` con el generador y el juez reales, `RUNS_PER_CASE` ejecuciones por caso, y afirma que la proporción de casos que pasan ≥ `PASS_RATE_THRESHOLD`.
- **Tarda ~30+ minutos en CPU** con `qwen2.5:3b`: 12 casos × 3 ejecuciones, con hasta 2 llamadas por ejecución (generador + juez; los casos que se abstienen solo llaman al juez).
- Si Ollama no responde o falta el modelo, la prueba se **salta limpiamente en segundos** (`skipped`, con el motivo), sin esperar al timeout de generación.
- El reporte Markdown de la ejecución se escribe en el `tmp_path` de pytest (su ruta se imprime con `-s`).

### Reporte antes/después

```powershell
.venv\Scripts\python -m rag_lab --out reports/report.md
```

Calcula las métricas deterministas en ambos modos (no necesita Ollama) y escribe `reports/report.md`. Salida esperada (en Windows; en Linux/macOS la ruta se imprime como `reports/report.md`):

```
Reporte escrito en reports\report.md
  Antes (shared): contaminación 41.7% (5/12)
  Después (isolated): contaminación 0.0% (0/12)
```

Con `--with-llm` añade la evaluación con juez en **ambos** modos (el doble de llamadas que `llm_eval`, así que en CPU cuenta con más de una hora). Antes de indexar comprueba que el LLM responde; si no, muestra un error claro y termina con código 1 sin escribir nada.

```powershell
.venv\Scripts\python -m rag_lab --with-llm --out reports/report.md
```

Los reportes generados (`reports/*.md`) no se versionan: los resultados de referencia están pegados abajo.

## Resultados antes/después

### Métricas deterministas (sin LLM)

Salida de `python -m rag_lab --out reports/report.md` (determinista: se reproduce exactamente).

| Métrica | Antes (shared) | Después (isolated) |
| --- | --- | --- |
| Tasa de contaminación | 41.7% (5/12) | 0.0% (0/12) |
| Exactitud de abstención | 83.3% (10/12) | 100.0% (12/12) |
| Casos contaminados | gs-06, gs-09, gs-10, gs-11, gs-12 | — |
| Casos con abstención incorrecta | gs-09, gs-10 | — |

> **Cómo se mide la contaminación.** Es una métrica determinista sobre la recuperación: un caso está contaminado si el top-k recuperado para su agente contiene algún chunk de otro agente; no la puntúa el juez. En modo shared una fidelidad alta del juez no implica ausencia de contaminación: el juez puntúa la respuesta contra el contexto contaminado que vio el generador, así que una respuesta copiada de un chunk ajeno puede ser perfectamente "fiel".

### Evaluación con juez LLM

Ejecución real del **2026-09-23**: `qwen2.5:3b` como generador y como juez, Ollama en Docker **en CPU**, modo `isolated`, 3 ejecuciones por caso, ~34 minutos (`pytest -m llm_eval`). **Estos resultados no son deterministas**: otra ejecución puede dar tasas distintas por caso. Resumen: **10/12 casos pasan (83.3%, umbral 0.67), 0 ejecuciones con error**. Fallan gs-02 y gs-03, dos preguntas de dos partes en las que el modelo responde solo una parte.

Tasa de éxito = ejecuciones aprobadas por el juez / ejecuciones del caso. La justificación corresponde a la primera ejecución con el mismo resultado que el caso.

#### Antes (shared)

_Sin resultados de evaluación LLM para este modo._ (`llm_eval` evalúa solo el modo `isolated`; el antes/después del bug se mide con las métricas deterministas de arriba.)

#### Después (isolated)

Casos que pasan: 83.3% (10/12) · Ejecuciones con error: 0 de 36

| Caso | Agente | Esperado | Tasa de éxito | Resultado | Justificación del juez |
| --- | --- | --- | --- | --- | --- |
| gs-01 | faq | answer | 100.0% (3/3) | ✅ pasa | faithfulness 1.00: The answer addresses both cost and delivery time, matching the context exactly. |
| gs-02 | faq | answer | 33.3% (1/3) | ❌ no pasa | faithfulness 0.50: The answer covers the time frame for returns but ignores return label costs. |
| gs-03 | faq | answer | 0.0% (0/3) | ❌ no pasa | relevance 0.33: The answer addresses the question about the reset link's validity, but does not cover what happens afte… |
| gs-04 | faq | answer | 100.0% (3/3) | ✅ pasa | faithfulness 0.90: The answer includes information about the warranty period for manufacturing defects, which matches t… |
| gs-05 | seguimiento | answer | 66.7% (2/3) | ✅ pasa | faithfulness 1.00: The answer covers the number of paid vacation days per year correctly. |
| gs-06 | seguimiento | answer | 66.7% (2/3) | ✅ pasa | faithfulness 0.70: The answer mentions the overtime pay rate (1.5 times base hourly rate) and the logging deadline, whi… |
| gs-07 | seguimiento | answer | 66.7% (2/3) | ✅ pasa | faithfulness 1.00: The answer covers all key points regarding parental leave entitlement and notification process. |
| gs-08 | seguimiento | answer | 100.0% (3/3) | ✅ pasa | faithfulness 1.00: The answer states the key information correctly. |
| gs-09 | faq | abstain | 100.0% (3/3) | ✅ pasa | abstention 1.00: The assistant declines to answer but provides a statement that aligns with expected behavior of declin… |
| gs-10 | seguimiento | abstain | 100.0% (3/3) | ✅ pasa | abstention 1.00: The assistant has clearly stated that it does not have the information and declined answering the ques… |
| gs-11 | faq | abstain | 100.0% (3/3) | ✅ pasa | abstention 1.00: The assistant SHOULD DECLINE to answer this specific weather-related question as it is outside their k… |
| gs-12 | seguimiento | abstain | 100.0% (3/3) | ✅ pasa | abstention 1.00: The assistant clearly indicated they don’t have the information and did not attempt to guess or give a… |

## Estructura

```
data/            corpus (faq_docs.json, seguimiento_docs.json) y golden_set.json
docs/plan.md     plan del proyecto y tareas
scripts/         calibrate_threshold.py (calibración de RELEVANCE_THRESHOLD)
src/rag_lab/     config, corpus, store, contamination, agent, llm, judge, evaluation, report, CLI (__main__)
tests/           pruebas deterministas (FakeLLM, Chroma en memoria); tests/llm_eval/ con el LLM real
reports/         reportes generados por la CLI (ignorados por git)
```

## Lecciones aprendidas
