import pytest

from src.metrics import precision_at_k, recall_at_k, reciprocal_rank

RELEVANTE = {"a.pdf"}


def test_precision_cuenta_solo_los_k_primeros():
    retrieved = ["a.pdf", "b.pdf", "a.pdf", "c.pdf", "a.pdf", "a.pdf"]
    assert precision_at_k(retrieved, RELEVANTE, 5) == pytest.approx(0.6)


def test_precision_divide_por_k_aunque_haya_menos_resultados():
    assert precision_at_k(["a.pdf"], RELEVANTE, 5) == pytest.approx(0.2)


def test_recall_es_1_si_el_relevante_aparece_en_top_k():
    assert recall_at_k(["b.pdf", "a.pdf"], RELEVANTE, 5) == 1.0


def test_recall_es_0_si_el_relevante_queda_fuera_de_top_k():
    assert recall_at_k(["b.pdf", "c.pdf", "a.pdf"], RELEVANTE, 2) == 0.0


def test_recall_con_varios_relevantes():
    assert recall_at_k(["a.pdf", "c.pdf"], {"a.pdf", "b.pdf"}, 5) == pytest.approx(0.5)


def test_reciprocal_rank():
    assert reciprocal_rank(["b.pdf", "c.pdf", "a.pdf"], RELEVANTE) == pytest.approx(1 / 3)
    assert reciprocal_rank(["b.pdf"], RELEVANTE) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_k_invalido(k):
    with pytest.raises(ValueError):
        precision_at_k(["a.pdf"], RELEVANTE, k)
    with pytest.raises(ValueError):
        recall_at_k(["a.pdf"], RELEVANTE, k)


def test_recall_sin_relevantes_es_error():
    with pytest.raises(ValueError):
        recall_at_k(["a.pdf"], set(), 5)
