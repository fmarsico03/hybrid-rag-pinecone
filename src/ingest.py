import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PATH,
    DATA_DIR,
    EMBED_MODEL,
    PINECONE_API_KEY,
    PINECONE_NAMESPACE,
)
from src.setup_index import ensure_index


def load_pdfs() -> list[Document]:
    """Carga los PDFs de data/ dejando en la metadata solo archivo y página (1-based)."""
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No hay PDFs en {DATA_DIR}.")

    docs = []
    for pdf_path in pdfs:
        paginas = PyPDFLoader(str(pdf_path)).load()
        for doc in paginas:
            doc.metadata = {"source": pdf_path.name, "page": doc.metadata["page"] + 1}
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


def ingest(force: bool = False) -> None:
    """Carga, fragmenta y sube los chunks. Si el namespace ya tiene vectores no reindexa."""
    vectorstore = get_vectorstore()
    stats = vectorstore.index.describe_index_stats()
    existentes = stats.namespaces.get(PINECONE_NAMESPACE)
    total = existentes.vector_count if existentes else 0

    if total > 0 and CHUNKS_PATH.exists() and not force:
        print(f"El namespace '{PINECONE_NAMESPACE}' ya tiene {total} vectores. No se reindexa.")
        return

    chunks = split_docs(load_pdfs())
    save_chunks(chunks)

    # Los ids son determinísticos, así que re-subir sobrescribe en lugar de duplicar.
    vectorstore.add_documents(chunks, ids=[c.id for c in chunks])
    print(f"Subidos {len(chunks)} chunks al namespace '{PINECONE_NAMESPACE}'.")


if __name__ == "__main__":
    import sys

    ingest(force="--force" in sys.argv)
