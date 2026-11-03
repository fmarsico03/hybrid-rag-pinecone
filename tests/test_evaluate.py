import json

import pytest
from langchain_core.documents import Document

from src.evaluate import evaluate_mode, load_golden_set


def _escribir(tmp_path, items):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def test_load_golden_set_valido(tmp_path):
    items = [{"pregunta": "¿?", "documento_id_esperado": "a.pdf"}]
    assert load_golden_set(_escribir(tmp_path, items), {"a.pdf"}) == items


def test_load_golden_set_rechaza_campos_faltantes(tmp_path):
    with pytest.raises(ValueError, match="faltan"):
        load_golden_set(_escribir(tmp_path, [{"pregunta": "¿?"}]), {"a.pdf"})


def test_load_golden_set_rechaza_documento_inexistente(tmp_path):
    items = [{"pregunta": "¿?", "documento_id_esperado": "z.pdf"}]
    with pytest.raises(ValueError, match="no está"):
        load_golden_set(_escribir(tmp_path, items), {"a.pdf"})


def test_evaluate_mode_calcula_metricas_por_pregunta():
    def fake_retrieve(_query):
        return [Document(page_content="", metadata={"source": s}) for s in ["b.pdf", "a.pdf", "a.pdf", "c.pdf", "d.pdf"]]

    [resultado] = evaluate_mode(fake_retrieve, [{"pregunta": "q", "documento_id_esperado": "a.pdf"}], k=5)

    assert resultado["precision"] == pytest.approx(0.4)
    assert resultado["recall"] == 1.0
    assert resultado["rr"] == pytest.approx(0.5)
