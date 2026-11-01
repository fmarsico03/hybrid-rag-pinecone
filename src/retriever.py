import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from src.config import TOP_K, WEIGHTS
from src.ingest import get_vectorstore, load_chunks


def tokenize(text: str) -> list[str]:
    """Minúsculas y solo palabras: sin esto 'Agent,' y 'agent' serían términos distintos para BM25."""
    return re.findall(r"\w+", text.lower())


def normalize_metadata(docs: list[Document]) -> list[Document]:
    """Pinecone devuelve los números como float (p. ej. page=5.0): los pasamos a int."""
    for doc in docs:
        if isinstance(doc.metadata.get("page"), float):
            doc.metadata["page"] = int(doc.metadata["page"])
    return docs


def get_bm25_retriever() -> BM25Retriever:
    """Retriever léxico (keywords), construido en memoria a partir de los chunks de la ingesta."""
    return BM25Retriever.from_documents(load_chunks(), k=TOP_K, preprocess_func=tokenize)


def get_dense_retriever():
    """Retriever semántico sobre Pinecone."""
    return get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})


def get_retriever() -> Runnable:
    """Retriever híbrido: combina BM25 y Pinecone con Reciprocal Rank Fusion según WEIGHTS."""
    ensemble = EnsembleRetriever(
        retrievers=[get_bm25_retriever(), get_dense_retriever()],
        weights=WEIGHTS,
    )
    return ensemble | RunnableLambda(lambda docs: normalize_metadata(docs)[:TOP_K])


def _print_docs(titulo: str, docs: list[Document]) -> None:
    print(f"\n--- {titulo} ---")
    for i, doc in enumerate(normalize_metadata(docs), 1):
        texto = doc.page_content[:120].replace("\n", " ")
        print(f"{i}. {doc.metadata['source']} p.{doc.metadata['page']} | {texto}")


if __name__ == "__main__":
    bm25 = get_bm25_retriever()
    dense = get_dense_retriever()
    hybrid = get_retriever()

    while True:
        pregunta = input("\nPregunta (enter para salir): ").strip()
        if not pregunta:
            break
        _print_docs("BM25 (keywords)", bm25.invoke(pregunta))
        _print_docs("Pinecone (semántico)", dense.invoke(pregunta))
        _print_docs(f"Híbrido (pesos {WEIGHTS})", hybrid.invoke(pregunta))
