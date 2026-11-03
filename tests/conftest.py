import os

# src.config exige estas variables al importarse. Los tests unitarios no llaman a ninguna API,
# así que alcanza con valores de prueba (setdefault respeta las reales si ya están definidas).
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("INDEX_NAME", "test-index")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
