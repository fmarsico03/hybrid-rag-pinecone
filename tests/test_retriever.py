from langchain_core.documents import Document

from src.retriever import normalize_metadata, tokenize


def test_tokenize_normaliza_mayusculas_y_puntuacion():
    assert tokenize("Agent, root_agent: ADK!") == ["agent", "root_agent", "adk"]


def test_tokenize_conserva_acentos():
    assert tokenize("Configuración") == ["configuración"]


def test_normalize_metadata_convierte_page_float_a_int():
    docs = [Document(page_content="x", metadata={"page": 5.0}), Document(page_content="y", metadata={"page": 2})]
    assert [d.metadata["page"] for d in normalize_metadata(docs)] == [5, 2]
    assert isinstance(docs[0].metadata["page"], int)
