"""Métricas de recuperación. Funciones puras: reciben la lista ordenada de ids recuperados
y el conjunto de ids relevantes, sin depender de Pinecone ni de LangChain."""


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fracción de los k primeros resultados que son relevantes."""
    if k <= 0:
        raise ValueError("k debe ser mayor a 0.")
    return sum(1 for doc_id in retrieved[:k] if doc_id in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fracción de los documentos relevantes que aparecen en los k primeros resultados."""
    if k <= 0:
        raise ValueError("k debe ser mayor a 0.")
    if not relevant:
        raise ValueError("relevant no puede estar vacío.")
    return len(relevant & set(retrieved[:k])) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / posición del primer resultado relevante (0 si no aparece ninguno)."""
    for posicion, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1 / posicion
    return 0.0
