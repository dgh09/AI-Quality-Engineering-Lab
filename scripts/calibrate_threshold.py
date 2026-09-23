"""Calibración de RELEVANCE_THRESHOLD: distancia top-1 por pregunta del golden set.

Indexa el corpus real en un cliente Chroma efímero propio y, para cada caso del
golden set, imprime la distancia del chunk más cercano en ambos modos:

- shared:   `query(..., agent_id=None)` (sin filtro, con el bug de contaminación).
- isolated: `query(..., agent_id=case.agent_id)` (filtro en la capa de datos).

Al final resume, en modo isolated, la distancia top-1 máxima de los casos que deben
responder y la mínima de los que deben abstenerse; el umbral debe caer entre ambas.

Uso: .venv\\Scripts\\python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import chromadb

from rag_lab.config import DEFAULT_TOP_K
from rag_lab.corpus import load_documents, load_golden_set
from rag_lab.store import VectorStore


def main() -> None:
    documents = load_documents()
    cases = load_golden_set(documents=documents)
    # Cliente efímero propio: no comparte estado con las pruebas ni con .chroma/.
    store = VectorStore(chromadb.EphemeralClient(), collection_name="calibration")
    store.index(documents)

    header = (
        f"{'case':<6} {'agent':<12} {'category':<14} {'expect':<8} "
        f"{'shared top-1':<22} {'isolated top-1':<22} isolated top-{DEFAULT_TOP_K}"
    )
    print(header)
    print("-" * len(header))
    answer_dists: dict[str, float] = {}
    abstain_dists: dict[str, float] = {}
    for case in cases:
        shared = store.query(case.question, k=DEFAULT_TOP_K, agent_id=None)
        isolated = store.query(case.question, k=DEFAULT_TOP_K, agent_id=case.agent_id)
        shared_top = f"{shared[0].distance:.4f} {shared[0].id}"
        isolated_top = f"{isolated[0].distance:.4f} {isolated[0].id}"
        top_k_ids = ",".join(chunk.id for chunk in isolated)
        print(
            f"{case.id:<6} {case.agent_id:<12} {case.category:<14} "
            f"{case.expected_behavior:<8} {shared_top:<22} {isolated_top:<22} {top_k_ids}"
        )
        target = answer_dists if case.expected_behavior == "answer" else abstain_dists
        target[case.id] = isolated[0].distance

    worst_answer = max(answer_dists, key=answer_dists.__getitem__)
    closest_abstain = min(abstain_dists, key=abstain_dists.__getitem__)
    lo, hi = answer_dists[worst_answer], abstain_dists[closest_abstain]
    print()
    print(f"isolated: max top-1 de 'answer'  = {lo:.4f} ({worst_answer})")
    print(f"isolated: min top-1 de 'abstain' = {hi:.4f} ({closest_abstain})")
    if lo < hi:
        print(f"separables: cualquier umbral en [{lo:.4f}, {hi:.4f}) separa los casos")
    else:
        print("NO separables: ningún umbral separa 'answer' de 'abstain' en modo isolated")


if __name__ == "__main__":
    main()
