from langchain_core.documents import Document

from src import ingest
from src.config import DEFAULT_CATEGORY

CATEGORIES = {"a.pdf": {"category": "setup", "tags": ["instalacion"]}}


def test_build_metadata_con_categoria():
    assert ingest.build_metadata("a.pdf", 3, CATEGORIES) == {
        "source": "a.pdf",
        "page": 3,
        "category": "setup",
        "tags": ["instalacion"],
    }


def test_build_metadata_sin_categoria_usa_default():
    metadata = ingest.build_metadata("otro.pdf", 1, CATEGORIES)
    assert metadata["category"] == DEFAULT_CATEGORY
    assert metadata["tags"] == []


def _pagina(texto: str, source: str = "a.pdf", page: int = 1) -> Document:
    return Document(page_content=texto, metadata=ingest.build_metadata(source, page, CATEGORIES))


def test_split_docs_asigna_ids_unicos_y_deterministicos():
    texto_largo = "palabra " * 600  # ~4800 caracteres: genera varios chunks
    docs = [_pagina(texto_largo, page=1), _pagina("corto", page=2), _pagina("otro", source="b.pdf")]

    chunks = ingest.split_docs(docs)
    ids = [c.id for c in chunks]

    assert len(ids) == len(set(ids))
    assert ids[0] == "a.pdf-p1-c1"
    assert "a.pdf-p2-c1" in ids
    assert "b.pdf-p1-c1" in ids
    assert ids == [c.id for c in ingest.split_docs(docs)]


def test_split_docs_conserva_la_metadata():
    chunks = ingest.split_docs([_pagina("hola mundo", page=4)])
    assert chunks[0].metadata["page"] == 4
    assert chunks[0].metadata["category"] == "setup"


def test_save_y_load_chunks_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHUNKS_PATH", tmp_path / "processed" / "chunks.json")
    chunks = ingest.split_docs([_pagina("texto con acentos: configuración")])

    ingest.save_chunks(chunks)
    cargados = ingest.load_chunks()

    assert [(c.id, c.page_content, c.metadata) for c in cargados] == [
        (c.id, c.page_content, c.metadata) for c in chunks
    ]
