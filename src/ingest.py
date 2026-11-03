import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CATEGORIES_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PATH,
    DATA_DIR,
    DEFAULT_CATEGORY,
    EMBED_MODEL,
    PINECONE_API_KEY,
    PINECONE_NAMESPACE,
)
from src.setup_index import ensure_index


def load_categories() -> dict[str, dict]:
    """Lee el mapeo archivo -> {category, tags}. Si no existe, todos los documentos quedan sin categoría."""
    if not CATEGORIES_PATH.exists():
        return {}
    return json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))


def build_metadata(source: str, page: int, categories: dict[str, dict]) -> dict:
    """Metadata de cada página: fuente, página (1-based), categoría y etiquetas."""
    info = categories.get(source, {})
    return {
        "source": source,
        "page": page,
        "category": info.get("category", DEFAULT_CATEGORY),
        "tags": list(info.get("tags", [])),
    }


def load_pdfs() -> list[Document]:
    """Carga los PDFs de data/ con la metadata definida en build_metadata."""
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No hay PDFs en {DATA_DIR}.")

    categories = load_categories()
    docs = []
    for pdf_path in pdfs:
        if pdf_path.name not in categories:
            print(f"Aviso: {pdf_path.name} no está en {CATEGORIES_PATH.name}, se usa '{DEFAULT_CATEGORY}'.")
        paginas = PyPDFLoader(str(pdf_path)).load()
        for doc in paginas:
            doc.metadata = build_metadata(pdf_path.name, doc.metadata["page"] + 1, categories)
        docs.extend(paginas)
        print(f"{pdf_path.name}: {len(paginas)} páginas")

    print(f"Total: {len(docs)} páginas")
    return docs


def split_docs(docs: list[Document]) -> list[Document]:
    """Fragmenta y asigna un id determinístico a cada chunk (archivo-página-n)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    contador: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        clave = (chunk.metadata["source"], chunk.metadata["page"])
        contador[clave] = contador.get(clave, 0) + 1
        chunk.id = f"{clave[0]}-p{clave[1]}-c{contador[clave]}"

    print(f"Total: {len(chunks)} chunks")
    return chunks


def save_chunks(chunks: list[Document]) -> None:
    """Persiste los chunks en disco: BM25 es local y los necesita para reconstruirse."""
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = [{"id": c.id, "text": c.page_content, "metadata": c.metadata} for c in chunks]
    CHUNKS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Chunks guardados en {CHUNKS_PATH}")


def load_chunks() -> list[Document]:
    """Lee los chunks persistidos por la ingesta."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"No existe {CHUNKS_PATH}. Corré primero: python -m src.ingest")
    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return [Document(id=d["id"], page_content=d["text"], metadata=d["metadata"]) for d in data]


def get_vectorstore() -> PineconeVectorStore:
    """Vector store sobre el índice, con los embeddings de Pinecone Inference."""
    embeddings = PineconeEmbeddings(model=EMBED_MODEL, pinecone_api_key=PINECONE_API_KEY)
    return PineconeVectorStore(
        index=ensure_index(),
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )


def namespace_vector_count(vectorstore: PineconeVectorStore) -> int:
    stats = vectorstore.index.describe_index_stats()
    existentes = stats.namespaces.get(PINECONE_NAMESPACE)
    return existentes.vector_count if existentes else 0


def ingest(force: bool = False) -> None:
    """Carga, fragmenta y sube los chunks. Si el namespace ya tiene vectores no reindexa.

    Con force=True borra el namespace antes de subir, para no dejar vectores huérfanos
    de documentos eliminados o de un chunking anterior.
    """
    vectorstore = get_vectorstore()
    total = namespace_vector_count(vectorstore)

    if total > 0 and CHUNKS_PATH.exists() and not force:
        print(f"El namespace '{PINECONE_NAMESPACE}' ya tiene {total} vectores. No se reindexa.")
        return

    chunks = split_docs(load_pdfs())
    save_chunks(chunks)

    if total > 0:
        vectorstore.index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print(f"Borrados {total} vectores del namespace '{PINECONE_NAMESPACE}'.")

    # Los ids son determinísticos, así que una re-subida parcial sobrescribe en lugar de duplicar.
    vectorstore.add_documents(chunks, ids=[c.id for c in chunks])
    print(f"Subidos {len(chunks)} chunks al namespace '{PINECONE_NAMESPACE}'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingesta de PDFs en Pinecone.")
    parser.add_argument("--force", action="store_true", help="Borra el namespace y reindexa todo.")
    ingest(force=parser.parse_args().force)
