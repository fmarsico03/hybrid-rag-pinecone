import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

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


class RAGSystem:
    """Recuperador híbrido: combina BM25 (léxico) y Pinecone (semántico) con un EnsembleRetriever.

    El EnsembleRetriever fusiona ambos rankings con Reciprocal Rank Fusion usando `weights`
    (en orden: BM25, semántico). `retrieve()` devuelve los `top_k` mejores documentos.
    """

    def __init__(self, top_k: int = TOP_K, weights: list[float] = WEIGHTS):
        if len(weights) != 2:
            raise ValueError("weights debe tener 2 valores: [peso_bm25, peso_semantico].")
        self.top_k = top_k
        self.weights = weights
        self.chunks = load_chunks()
        self.categories = sorted({c.metadata["category"] for c in self.chunks})
        self._vectorstore = get_vectorstore()
        # Un ensemble por categoría (None = sin filtro), construido la primera vez que se pide.
        self._ensembles: dict[str | None, EnsembleRetriever] = {}

    def _bm25(self, category: str | None = None) -> BM25Retriever:
        chunks = [c for c in self.chunks if category is None or c.metadata["category"] == category]
        return BM25Retriever.from_documents(chunks, k=self.top_k, preprocess_func=tokenize)

    def _dense(self, category: str | None = None) -> BaseRetriever:
        search_kwargs: dict = {"k": self.top_k}
        if category is not None:
            search_kwargs["filter"] = {"category": {"$eq": category}}
        return self._vectorstore.as_retriever(search_kwargs=search_kwargs)

    def _check_category(self, category: str | None) -> None:
        if category is not None and category not in self.categories:
            raise ValueError(f"Categoría desconocida '{category}'. Disponibles: {self.categories}")

    def ensemble(self, category: str | None = None) -> EnsembleRetriever:
        self._check_category(category)
        if category not in self._ensembles:
            self._ensembles[category] = EnsembleRetriever(
                retrievers=[self._bm25(category), self._dense(category)],
                weights=self.weights,
            )
        return self._ensembles[category]

    def retrieve(self, query: str, category: str | None = None) -> list[Document]:
        """Top-k híbrido para `query`, opcionalmente filtrado por categoría."""
        return normalize_metadata(self.ensemble(category).invoke(query))[: self.top_k]

    def retrieve_bm25(self, query: str) -> list[Document]:
        """Solo búsqueda léxica (para comparar en la evaluación)."""
        return self.ensemble().retrievers[0].invoke(query)[: self.top_k]

    def retrieve_dense(self, query: str) -> list[Document]:
        """Solo búsqueda semántica (para comparar en la evaluación)."""
        return normalize_metadata(self.ensemble().retrievers[1].invoke(query))[: self.top_k]


def _print_docs(titulo: str, docs: list[Document]) -> None:
    print(f"\n--- {titulo} ---")
    for i, doc in enumerate(docs, 1):
        texto = doc.page_content[:120].replace("\n", " ")
        print(f"{i}. [{doc.metadata['category']}] {doc.metadata['source']} p.{doc.metadata['page']} | {texto}")


if __name__ == "__main__":
    rag_system = RAGSystem()

    while True:
        pregunta = input("\nPregunta (enter para salir): ").strip()
        if not pregunta:
            break
        _print_docs("BM25 (keywords)", rag_system.retrieve_bm25(pregunta))
        _print_docs("Pinecone (semántico)", rag_system.retrieve_dense(pregunta))
        _print_docs(f"Híbrido (pesos {rag_system.weights})", rag_system.retrieve(pregunta))
