# RAG híbrido con Pinecone (BM25 + búsqueda semántica)

Flujo RAG end-to-end con **recuperación híbrida**: cada pregunta se busca en paralelo por palabras clave (BM25) y por significado (embeddings en Pinecone), y ambos rankings se fusionan antes de pasarle el contexto al LLM. El modelo responde **solo** con la información recuperada; si la respuesta no está en los documentos, responde `"No lo sé"`.

- **LLM:** Google Gemini (`gemini-3.8-flash`), vía `langchain-google-genai`
- **Embeddings:** `multilingual-e5-large` (1024 dim) vía Pinecone Inference
- **Base vectorial:** Pinecone serverless (índice `hybrid-rag`, namespace `docs`)
- **Búsqueda léxica:** BM25 en memoria (`rank-bm25`)
- **Orquestación:** LangChain (LCEL) + `EnsembleRetriever` + `PydanticOutputParser`

## ¿Por qué híbrido?

| Búsqueda | Es buena para | Falla con |
|---|---|---|
| **BM25 (keywords)** | Términos exactos: nombres de funciones, siglas, comandos (`root_agent`, `adk run`) | Sinónimos, paráfrasis, preguntas en otro idioma |
| **Semántica (Pinecone)** | Preguntas conceptuales y reformuladas, multilingüe | Identificadores raros o términos muy específicos |

El retriever híbrido combina ambos con **Reciprocal Rank Fusion (RRF)**: cada documento suma `peso / (60 + posición)` por cada ranking en el que aparece. Como usa posiciones y no scores, no importa que BM25 y coseno estén en escalas distintas. Un fragmento que aparece arriba en las dos listas queda primero.

## Estructura

```
├── data/
│   ├── *.pdf               # Dataset de ejemplo (4 PDFs de apuntes del curso sobre agentes / ADK)
│   └── processed/
│       └── chunks.json     # Chunks generados por la ingesta (se genera, no se versiona)
├── src/
│   ├── config.py           # Configuración desde .env + parámetros de chunking y retrieval
│   ├── setup_index.py      # Crea el índice de Pinecone si no existe y valida la dimensión
│   ├── ingest.py           # Carga, chunking, persistencia local y upsert a Pinecone
│   ├── retriever.py        # Retrievers BM25, semántico e híbrido + modo comparativo
│   ├── rag.py              # Cadena RAG asíncrona + pruebas
│   └── main.py             # Punto de entrada: ingesta + preguntas interactivas
├── .env.example            # Plantilla de variables de entorno
└── requirements.txt
```

## Cómo funciona

1. **Índice (`setup_index.py`)**
   - Crea un índice serverless en Pinecone con dimensión 1024 y métrica coseno, y espera a que esté listo.
   - Si el índice ya existe, verifica que su dimensión coincida con la del modelo de embeddings.

2. **Ingesta (`ingest.py`)**
   - Lee los PDFs de `data/` con `PyPDFLoader` y guarda en la metadata el archivo y la página.
   - Fragmenta con `RecursiveCharacterTextSplitter`: chunks de **1600 caracteres con 200 de overlap**. `multilingual-e5-large` trunca a 512 tokens, y 1600 caracteres entran con margen.
   - Asigna a cada chunk un **id determinístico** (`archivo.pdf-p5-c2`): re-subir los documentos sobrescribe en lugar de duplicar.
   - Guarda los chunks en `data/processed/chunks.json`. **BM25 es local** y necesita el corpus para reconstruirse sin volver a ingerir.
   - Sube los chunks a Pinecone. `PineconeEmbeddings` agrega solo los prefijos `passage:` / `query:` que espera el modelo e5.
   - Si el namespace ya tiene vectores, no vuelve a indexar (salvo con `--force`).

3. **Recuperación (`retriever.py`)**
   - `get_bm25_retriever()`: BM25 sobre los chunks de `chunks.json`, con un tokenizador que pasa a minúsculas y descarta la puntuación.
   - `get_dense_retriever()`: búsqueda por similitud en Pinecone.
   - `get_retriever()`: `EnsembleRetriever` con RRF y pesos `WEIGHTS = [0.5, 0.5]` (BM25, semántico). Elimina duplicados y devuelve los `TOP_K = 5` mejores.
   - Ejecutado directamente, abre un modo interactivo que muestra **los tres rankings lado a lado** para cada pregunta.

4. **Generación grounded (`rag.py`)**
   - Cadena LCEL: `retriever híbrido | format_docs → prompt → Gemini → PydanticOutputParser`.
   - El prompt de sistema funciona como filtro de veracidad: responder solo con el CONTEXTO y, si no está, decir `"No lo sé"`.
   - La función asíncrona `get_rag_response(query: str)` devuelve un modelo Pydantic:

     ```python
     class RAGResponse(BaseModel):
         respuesta: str          # texto de la respuesta o "No lo sé"
         referencias: list[str]  # p. ej. ["archivo.pdf p.2"]; vacía si no sabe
     ```

5. **Punto de entrada (`main.py`)**
   - Ejecuta la ingesta (que se saltea si ya está hecha) y abre un loop de preguntas.

## Instalación

Requiere Python 3.11+ (probado con 3.13).

```bash
# 1. Crear y activar el entorno virtual
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env    # en Windows: copy .env.example .env
# Editar .env y completar PINECONE_API_KEY y GOOGLE_API_KEY
```

- La API key de Pinecone se obtiene en [app.pinecone.io](https://app.pinecone.io) (el plan gratuito alcanza).
- La API key de Google se obtiene en [Google AI Studio](https://aistudio.google.com/apikey).

El archivo `.env` está en el `.gitignore` y no se sube al repositorio.

### Variables de entorno

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `PINECONE_API_KEY` | Sí | — | API key de Pinecone |
| `INDEX_NAME` | Sí | — | Nombre del índice (p. ej. `hybrid-rag`) |
| `PINECONE_NAMESPACE` | No | `docs` | Namespace donde se guardan los vectores |
| `CLOUD` / `REGION` | No | `aws` / `us-east-1` | Ubicación del índice serverless |
| `EMBED_MODEL` | No | `multilingual-e5-large` | Modelo de embeddings de Pinecone Inference |
| `GOOGLE_API_KEY` | Sí | — | API key de Gemini |
| `LLM_MODEL` | No | `gemini-3.8-flash` | Modelo de Gemini para generar las respuestas |

## Ejecución

Los scripts se ejecutan **como módulos desde la raíz del proyecto**:

```bash
# Opción rápida: indexa (si hace falta) y abre el loop de preguntas
python -m src.main
```

O paso a paso:

```bash
# 1. (Opcional) Crear el índice y validar las credenciales de Pinecone
python -m src.setup_index

# 2. Indexar los documentos (solo la primera vez; luego detecta los vectores existentes)
python -m src.ingest

# 3. (Opcional) Comparar BM25 vs. semántico vs. híbrido de forma interactiva
python -m src.retriever

# 4. Ejecutar la cadena RAG con las dos pruebas
python -m src.rag
```

Para volver a indexar (por ejemplo, si cambian los documentos o el tamaño de chunk):

```bash
python -m src.ingest --force
```

> Si se eliminan o renombran PDFs, `--force` sube los chunks nuevos pero no borra los viejos de Pinecone. En ese caso, borrar el namespace desde la consola de Pinecone antes de reindexar.

## Ejemplo de salida

`python -m src.rag`:

```
Pregunta: What Python version does ADK require?
Respuesta: ADK requires Python 3.11 or higher (Python 3.11+).
Referencias: ['T-DEVAGENT-I-m1-l1-en-file-1.en.pdf p.1']

Pregunta: What is the capital of Australia?
Respuesta: No lo sé
Referencias: []
```

`python -m src.main`, preguntando en español sobre documentos en inglés:

```
Pregunta (enter para salir): ¿Por qué el agente principal se llama root_agent?
Respuesta: Las herramientas de línea de comandos de ADK buscan una variable de Python llamada `root_agent` como punto de entrada a su sistema de agentes. Esta convención permite que ADK descubra y ejecute el agente, siendo su definición en el archivo `agent.py` el único elemento obligatorio de un agente de ADK.
Referencias: ['T-DEVAGENT-I-m2-l0-en-file-2.en.pdf p.5']
```

## Limitaciones y posibles mejoras

- **Preguntas en otro idioma:** BM25 compara palabras exactas, así que con una pregunta en español sobre documentos en inglés casi no aporta y el resultado depende del retriever semántico (que sí es multilingüe).
- **BM25 en memoria:** se reconstruye en cada ejecución a partir de `chunks.json`. Para corpus grandes convendría usar *sparse vectors* en Pinecone (p. ej. `pinecone-sparse-english-v0`) y hacer la búsqueda híbrida del lado del servidor.
- **Pesos fijos:** `WEIGHTS` está en `[0.5, 0.5]`; se puede ajustar en `config.py` según qué tipo de preguntas predomine.
