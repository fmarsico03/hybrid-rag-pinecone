import os
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    """Devuelve la variable de entorno `name` o aborta con un mensaje claro."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno requerida '{name}'. "
            "Definila en tu archivo .env (ver .env.example)."
        )
    return value


# --- Pinecone ---
PINECONE_API_KEY = _require("PINECONE_API_KEY")
INDEX_NAME = _require("INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "docs").strip() or "docs"

CLOUD = os.getenv("CLOUD", "aws").strip() or "aws"
REGION = os.getenv("REGION", "us-east-1").strip() or "us-east-1"

EMBED_MODEL = os.getenv("EMBED_MODEL", "multilingual-e5-large").strip() or "multilingual-e5-large"
EMBED_DIM = 1024

# --- Chunking ---
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 300

# --- Retrieval ---
TOP_K = 5
WEIGHTS = [0.5, 0.5]
