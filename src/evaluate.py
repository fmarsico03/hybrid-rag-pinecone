"""Evalúa la recuperación contra el golden set: Precision@k, Recall@k y MRR.

Cada pregunta del golden set indica el documento (PDF) que contiene la respuesta.
Un chunk recuperado se considera relevante si proviene de ese documento:
- Recall@k:    ¿aparece el documento esperado entre los k recuperados? (1 o 0)
- Precision@k: ¿qué fracción de los k recuperados viene del documento esperado?
- MRR:         1 / posición del primer chunk del documento esperado.

Se evalúan los tres modos (BM25, semántico e híbrido) para mostrar qué aporta cada uno.
"""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Callable

from langchain_core.documents import Document

from src.config import GOLDEN_SET_PATH, TOP_K
from src.metrics import precision_at_k, recall_at_k, reciprocal_rank
from src.retriever import RAGSystem


def load_golden_set(path: Path, fuentes_validas: set[str]) -> list[dict]:
    """Lee el golden set y valida que cada documento esperado exista en el corpus indexado."""
    items = json.loads(path.read_text(encoding="utf-8"))
    for i, item in enumerate(items, 1):
        if not {"pregunta", "documento_id_esperado"} <= item.keys():
            raise ValueError(f"Item {i} del golden set: faltan 'pregunta' o 'documento_id_esperado'.")
        if item["documento_id_esperado"] not in fuentes_validas:
            raise ValueError(f"Item {i}: '{item['documento_id_esperado']}' no está en los documentos indexados.")
    return items


def evaluate_mode(
    retrieve: Callable[[str], list[Document]], golden_set: list[dict], k: int
) -> list[dict]:
    """Corre cada pregunta con `retrieve` y calcula las métricas por pregunta."""
    resultados = []
    for item in golden_set:
        fuentes = [doc.metadata["source"] for doc in retrieve(item["pregunta"])]
        relevantes = {item["documento_id_esperado"]}
        resultados.append({
            "pregunta": item["pregunta"],
            "esperado": item["documento_id_esperado"],
            "recuperados": fuentes,
            "precision": precision_at_k(fuentes, relevantes, k),
            "recall": recall_at_k(fuentes, relevantes, k),
            "rr": reciprocal_rank(fuentes, relevantes),
        })
    return resultados


def _short(source: str) -> str:
    """'T-DEVAGENT-I-m1-l1-en-file-1.en.pdf' -> 'file-1'."""
    return source.split("-en-")[-1].removesuffix(".en.pdf")


def print_detail(nombre: str, resultados: list[dict], k: int) -> None:
    print(f"\nDetalle por pregunta — {nombre}")
    print("-" * 100)
    for i, r in enumerate(resultados, 1):
        top = ", ".join(_short(s) for s in r["recuperados"])
        print(f"Q{i}. {r['pregunta']}")
        print(
            f"    esperado: {_short(r['esperado'])} | top-{k}: [{top}] | "
            f"P@{k}={r['precision']:.2f}  R@{k}={r['recall']:.0f}  RR={r['rr']:.2f}"
        )


def print_summary(resumen: dict[str, list[dict]], k: int) -> None:
    print(f"\nResumen ({len(next(iter(resumen.values())))} preguntas, k={k})")
    print("-" * 52)
    print(f"{'Modo':<22}{f'Precision@{k}':>12}{f'Recall@{k}':>10}{'MRR':>8}")
    for nombre, resultados in resumen.items():
        p = mean(r["precision"] for r in resultados)
        rc = mean(r["recall"] for r in resultados)
        mrr = mean(r["rr"] for r in resultados)
        print(f"{nombre:<22}{p:>12.2f}{rc:>10.2f}{mrr:>8.2f}")


def main(k: int = TOP_K, golden_path: Path = GOLDEN_SET_PATH) -> dict[str, list[dict]]:
    rag_system = RAGSystem(top_k=k)
    golden_set = load_golden_set(golden_path, {c.metadata["source"] for c in rag_system.chunks})

    resumen = {
        "BM25 (léxico)": evaluate_mode(rag_system.retrieve_bm25, golden_set, k),
        "Pinecone (semántico)": evaluate_mode(rag_system.retrieve_dense, golden_set, k),
        f"Híbrido {rag_system.weights}": evaluate_mode(rag_system.retrieve, golden_set, k),
    }

    print_detail(f"Híbrido {rag_system.weights}", list(resumen.values())[-1], k)
    print_summary(resumen, k)
    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación de la recuperación con el golden set.")
    parser.add_argument("-k", type=int, default=TOP_K, help=f"Cantidad de documentos recuperados (default {TOP_K}).")
    parser.add_argument("--golden", type=Path, default=GOLDEN_SET_PATH, help="Ruta al golden set JSON.")
    args = parser.parse_args()
    main(k=args.k, golden_path=args.golden)
