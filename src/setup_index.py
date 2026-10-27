import time

from pinecone import Pinecone, ServerlessSpec

from src.config import CLOUD, EMBED_DIM, INDEX_NAME, PINECONE_API_KEY, REGION


def ensure_index(timeout: int = 120):
    """Crea el índice si no existe y devuelve el handle."""
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creando índice '{INDEX_NAME}' (dim={EMBED_DIM}, cosine)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        start = time.time()
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            if time.time() - start > timeout:
                raise TimeoutError(f"El índice '{INDEX_NAME}' no quedó listo en {timeout}s.")
            time.sleep(2)
        print(f"Índice '{INDEX_NAME}' listo.")
    else:
        dimension = pc.describe_index(INDEX_NAME).dimension
        if dimension != EMBED_DIM:
            raise RuntimeError(
                f"El índice '{INDEX_NAME}' tiene dimensión {dimension}, pero EMBED_DIM={EMBED_DIM}. "
                "Borrá el índice o usá otro INDEX_NAME."
            )
        print(f"Índice '{INDEX_NAME}' ya existe (dim={dimension}). OK.")

    return pc.Index(INDEX_NAME)


if __name__ == "__main__":
    ensure_index()
