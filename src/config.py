import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.json"


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

# --- LLM (Google Gemini) ---
GOOGLE_API_KEY = _require("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"

# --- Chunking ---
# En caracteres. multilingual-e5-large trunca a 512 tokens: ~1600 caracteres entran holgados.
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200

# --- Retrieval ---
TOP_K = 5
WEIGHTS = [0.5, 0.5]
