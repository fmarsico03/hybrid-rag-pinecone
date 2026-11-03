# RAG híbrido escalable con Pinecone (BM25 + búsqueda semántica)

Módulo de recuperación escalable en la nube: los documentos se indexan en **Pinecone Serverless** con metadatos (fuente, página, categoría y etiquetas), y cada consulta se resuelve con un **recuperador híbrido** que combina búsqueda léxica (BM25) y semántica (embeddings). La calidad de la recuperación se mide con **Precision@k, Recall@k y MRR** sobre un golden set. Encima, una cadena RAG con Gemini responde **solo** con la información recuperada; si la respuesta no está en los documentos, responde `"No lo sé"`.

- **Base vectorial:** Pinecone Serverless (índice `hybrid-rag`, namespace `docs`, métrica coseno)
- **Embeddings:** `multilingual-e5-large` (1024 dim) vía Pinecone Inference
- **Búsqueda léxica:** BM25 en memoria (`rank-bm25`)
- **Fusión:** `EnsembleRetriever` de LangChain (Reciprocal Rank Fusion)
- **LLM:** Google Gemini (`gemini-3.8-flash`), vía `langchain-google-genai`
- **Orquestación:** LangChain (LCEL) + `PydanticOutputParser`

## Cumplimiento de la consigna

| Requisito | Implementación |
|---|---|
| `.env` con API key de Pinecone, del LLM e `INDEX_NAME` | `.env.example`. Se usa Gemini como LLM (`GOOGLE_API_KEY`) |
| Script que verifica si el índice existe y lo crea (Serverless) | `src/setup_index.py` |
| Ingesta con `RecursiveCharacterTextSplitter` y embeddings en Pinecone | `src/ingest.py` |
| Texto original y fuente en la metadata de Pinecone | Campo `text` + `source`, `page`, `category`, `tags` |
| Namespaces | Todo vive en el namespace `PINECONE_NAMESPACE` (`docs`) |
| `BM25Retriever` + Pinecone combinados con `EnsembleRetriever` | `src/retriever.py` |
| Clase `RAGSystem` que devuelve el top-5 híbrido | `RAGSystem.retrieve(query)` en `src/retriever.py` |
| Golden set `{"pregunta", "documento_id_esperado"}` con 5 preguntas | `data/golden_set.json` |
| `evaluate.py` con Precision@5 y Recall@5 + reporte en consola | `src/evaluate.py` (además calcula MRR y compara los 3 modos) |
| Métricas resumidas en el README | [Resultados de la evaluación](#resultados-de-la-evaluación) |

## ¿Por qué híbrido?

| Búsqueda | Es buena para | Falla con |
|---|---|---|
| **BM25 (léxico)** | Términos exactos: nombres de clases, siglas, comandos (`InMemorySessionService`, `adk api_server`) | Sinónimos, paráfrasis, preguntas en otro idioma |
| **Semántica (Pinecone)** | Preguntas conceptuales y reformuladas, multilingüe | Identificadores raros o términos muy específicos |

El `EnsembleRetriever` combina ambos con **Reciprocal Rank Fusion (RRF)**: cada documento suma `peso / (60 + posición)` por cada ranking en el que aparece. Como usa posiciones y no scores, no importa que BM25 y coseno estén en escalas distintas, y un fragmento que aparece arriba en las dos listas queda primero.

## Estructura

```
├── data/
│   ├── *.pdf               # Dataset: 4 PDFs de documentación técnica de Google ADK
│   ├── categories.json     # Mapeo PDF -> categoría y etiquetas (metadata)
│   ├── golden_set.json     # 5 preguntas con su documento fuente esperado
│   └── processed/
│       └── chunks.json     # Chunks generados por la ingesta (no se versiona)
├── src/
│   ├── config.py           # Configuración desde .env + parámetros de chunking y retrieval
│   ├── setup_index.py      # Crea el índice serverless si no existe y valida la dimensión
│   ├── ingest.py           # Carga, chunking, metadata, persistencia local y upsert a Pinecone
│   ├── retriever.py        # Clase RAGSystem (BM25 + Pinecone + EnsembleRetriever)
│   ├── metrics.py          # Precision@k, Recall@k, reciprocal rank (funciones puras)
│   ├── evaluate.py         # Evaluación contra el golden set
│   ├── rag.py              # Cadena RAG asíncrona con Gemini + pruebas
│   └── main.py             # Punto de entrada: ingesta + preguntas interactivas
├── tests/                  # Tests unitarios (pytest, sin llamadas a APIs)
├── .env.example            # Plantilla de variables de entorno
├── requirements.txt        # Dependencias (versiones fijas)
└── requirements-dev.txt    # + pytest
```

## Cómo funciona

### 1. Índice (`setup_index.py`)
- Crea un índice **serverless** con dimensión 1024 y métrica coseno, y espera a que esté listo.
- Si el índice ya existe, valida que su dimensión coincida con la del modelo de embeddings. Así se evita el error de *dimension mismatch*.

### 2. Ingesta (`ingest.py`)
- Lee los PDFs de `data/` con `PyPDFLoader`.
- **Metadata** de cada chunk:

  ```json
  {
    "source": "T-DEVAGENT-I-m3-l0-en-file-4.en.pdf",
    "page": 3,
    "category": "deployment",
    "tags": ["adk-run", "api-server", "runner", "sesiones"],
    "text": "<texto original del chunk>"
  }
  ```

  La categoría y las etiquetas salen de `data/categories.json`. Un PDF que no figure ahí recibe la categoría `general`. El texto original queda en la metadata (`text`), así que una búsqueda en Pinecone devuelve el contenido sin consultar otra base de datos.
- Fragmenta con `RecursiveCharacterTextSplitter` en chunks de **1600 caracteres con 200 de overlap** (ver [Decisiones de diseño](#decisiones-de-diseño)).
- Asigna a cada chunk un **id determinístico** (`archivo.pdf-p5-c2`).
- Guarda los chunks en `data/processed/chunks.json`. **BM25 es local** y necesita el corpus para reconstruirse sin volver a ingerir.
- Sube los chunks a Pinecone. `PineconeEmbeddings` agrega solo los prefijos `passage:` / `query:` que espera el modelo e5.
- Si el namespace ya tiene vectores, no reindexa. Con `--force` borra el namespace y vuelve a subir todo, así no quedan vectores huérfanos de documentos eliminados.

### 3. Recuperación (`retriever.py`): la clase `RAGSystem`

```python
from src.retriever import RAGSystem

rag_system = RAGSystem()                        # top_k=5, weights=[0.5, 0.5]
docs = rag_system.retrieve("How do I run my agent as a REST API?")  # top-5 híbrido
docs = rag_system.retrieve("adk api_server", category="deployment")  # filtrado por metadata
```

- Encapsula un `EnsembleRetriever` con dos retrievers: `BM25Retriever` (sobre `chunks.json`, con un tokenizador que pasa a minúsculas y descarta la puntuación) y el retriever de `PineconeVectorStore`.
- `retrieve(query, category=None)` devuelve los `TOP_K = 5` mejores documentos combinando resultados léxicos y semánticos. Con `category` filtra **ambos** retrievers: Pinecone usa un filtro de metadata (`{"category": {"$eq": ...}}`) y BM25 se construye solo con los chunks de esa categoría.
- `retrieve_bm25()` y `retrieve_dense()` exponen cada mitad por separado, para la evaluación.
- `python -m src.retriever` abre un modo interactivo que muestra **los tres rankings lado a lado**.

### 4. Evaluación (`evaluate.py`)
- `data/golden_set.json` tiene 5 preguntas con el PDF que contiene la respuesta (`documento_id_esperado`).
- Para cada pregunta se recuperan los top-k chunks y se compara su `source` con el documento esperado:
  - **Recall@k:** ¿aparece el documento correcto entre los k recuperados? (1 o 0)
  - **Precision@k:** ¿qué fracción de los k recuperados viene del documento correcto?
  - **MRR** (extra): 1 / posición del primer chunk correcto, promediado.
- Se evalúan los tres modos (BM25, semántico e híbrido) y se imprime el detalle por pregunta del híbrido y un resumen comparativo.

### 5. Generación grounded (`rag.py`)
- Cadena LCEL: `RAGSystem.retrieve | format_docs → prompt → Gemini → PydanticOutputParser`.
- El prompt de sistema funciona como filtro de veracidad: responder solo con el CONTEXTO y, si no está, decir `"No lo sé"`.
- La función asíncrona `get_rag_response(query: str)` devuelve un modelo Pydantic:

  ```python
  class RAGResponse(BaseModel):
      respuesta: str          # texto de la respuesta o "No lo sé"
      referencias: list[str]  # p. ej. ["archivo.pdf p.2"]; vacía si no sabe
  ```

## Instalación

Requiere Python 3.11+ (probado con 3.13).

```bash
# 1. Crear y activar el entorno virtual
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 2. Instalar dependencias (requirements-dev.txt incluye pytest)
pip install -r requirements-dev.txt

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

## Replicar el índice de Pinecone

Todos los comandos se ejecutan **como módulos desde la raíz del proyecto**.

```bash
# 1. Crear el índice serverless (dim 1024, coseno) y validar las credenciales
python -m src.setup_index
# -> Creando índice 'hybrid-rag' (dim=1024, cosine)...
# -> Índice 'hybrid-rag' listo.

# 2. Procesar los PDFs y subirlos al namespace
python -m src.ingest
# -> Total: 28 páginas / Total: 33 chunks
# -> Subidos 33 chunks al namespace 'docs'.

# 3. Evaluar la recuperación
python -m src.evaluate          # k=5 por defecto; también: python -m src.evaluate -k 3
```

`ingest` también crea el índice si no existe, así que el paso 1 es opcional. Para reindexar desde cero (por ejemplo, si cambian los documentos o el chunking):

```bash
python -m src.ingest --force
```

Para usar otros documentos: copiarlos a `data/`, agregar su categoría en `data/categories.json`, actualizar `data/golden_set.json` y correr `python -m src.ingest --force`.

## Ejecución

```bash
python -m src.main        # ingesta (si hace falta) + loop de preguntas con respuesta de Gemini
python -m src.retriever   # comparar BM25 / semántico / híbrido de forma interactiva
python -m src.rag         # cadena RAG con dos pruebas (una respondible y una trampa)
python -m src.evaluate    # métricas de recuperación
python -m pytest          # tests unitarios (no necesitan API keys)
```

## Resultados de la evaluación

Salida de `python -m src.evaluate` (k=5):

```
Detalle por pregunta — Híbrido [0.5, 0.5]
----------------------------------------------------------------------------------------------------
Q1. What Python version does ADK require?
    esperado: file-1 | top-5: [file-1, file-1, file-1, file-1, file-4] | P@5=0.80  R@5=1  RR=1.00
Q2. Which command scaffolds a new agent project and which files does it create?
    esperado: file-1 | top-5: [file-1, file-4, file-1, file-2, file-1] | P@5=0.60  R@5=1  RR=1.00
Q3. What are the four core parameters that define an LlmAgent?
    esperado: file-2 | top-5: [file-2, file-2, file-2, file-3, file-3] | P@5=0.60  R@5=1  RR=1.00
Q4. Which example questions should I use to test the math tutor in the web interface?
    esperado: file-3 | top-5: [file-3, file-3, file-3, file-4, file-3] | P@5=0.80  R@5=1  RR=1.00
Q5. How do I run my agent as a REST API with adk api_server?
    esperado: file-4 | top-5: [file-4, file-4, file-4, file-3, file-1] | P@5=0.60  R@5=1  RR=1.00

Resumen (5 preguntas, k=5)
----------------------------------------------------
Modo                   Precision@5  Recall@5     MRR
BM25 (léxico)                 0.64      1.00    1.00
Pinecone (semántico)          0.76      1.00    1.00
Híbrido [0.5, 0.5]            0.68      1.00    1.00
```

Con k=3 (`python -m src.evaluate -k 3`): Precision@3 de 0.73 (BM25), 0.93 (semántico) y 0.87 (híbrido); Recall@3 y MRR de 1.00 en los tres.

**Lectura de los resultados:**
- **Recall@5 = 1.00 y MRR = 1.00 en los tres modos:** para las 5 preguntas, el documento correcto aparece y además queda **primero**. Con un corpus de 4 documentos temáticamente separados, las 5 preguntas no alcanzan para diferenciar los modos en recall.
- **Precision@5:** el semántico obtiene la mejor (0.76) y el híbrido queda entre los dos (0.68). BM25 trae chunks de otros documentos que comparten palabras (`adk web`, `agent`, `root_agent` aparecen en casi todos los PDFs), y la fusión hereda parte de ese ruido en las últimas posiciones.
- **Dónde aporta el híbrido:** en consultas con identificadores exactos. Con `What does the Runner do with InMemorySessionService?`, solo dos chunks contienen ese término (`file-4` p.4 y p.5). El semántico ubica p.4 en el **5° lugar**; BM25 la pone 2°, y el híbrido deja **ambos chunks en los dos primeros puestos**. Para un corpus con mayoría de preguntas conceptuales, los pesos se pueden inclinar hacia el semántico (p. ej. `WEIGHTS = [0.3, 0.7]` en `config.py`).

## Decisiones de diseño

- **Tamaño de chunk (1600 caracteres, ~265 tokens en promedio):** la consigna sugiere ~500–800 tokens, pero `multilingual-e5-large` solo acepta **512 tokens** por entrada y trunca el resto. Chunks más grandes perderían su final al generar el embedding. Se eligió el mayor tamaño que entra con margen (el chunk más grande mide ~500 tokens). Para usar chunks de 500–800 tokens habría que cambiar a un modelo de contexto más largo, como `llama-text-embed-v2` (2048 tokens, también 1024 dim).
- **Dimensión 1024 (no 1536):** la consigna menciona 1536 para `text-embedding-3-small` de OpenAI. Acá los embeddings se generan con Pinecone Inference (`multilingual-e5-large`, 1024 dim), así que no hace falta una API key de OpenAI y el índice se crea con 1024. `setup_index.py` valida que ambas coincidan.
- **LLM:** se usa Gemini en lugar de OpenAI o Anthropic. Solo interviene en la generación (`rag.py`); la ingesta, la recuperación y la evaluación no dependen del LLM.
- **Ids determinísticos:** re-subir los mismos documentos sobrescribe los vectores en lugar de duplicarlos.
- **Relevancia a nivel documento:** el golden set indica el PDF fuente (como pide la consigna), así que un chunk cuenta como relevante si viene de ese PDF.

## Limitaciones y posibles mejoras

- **Preguntas en otro idioma:** BM25 compara palabras exactas, así que una pregunta en español sobre documentos en inglés casi no recibe aporte léxico. El resultado depende del retriever semántico, que sí es multilingüe.
- **BM25 en memoria:** se reconstruye en cada ejecución a partir de `chunks.json`. Para corpus grandes convendría usar *sparse vectors* en Pinecone (p. ej. `pinecone-sparse-english-v0`) y hacer la búsqueda híbrida del lado del servidor.
- **Golden set chico:** 5 preguntas sobre 4 documentos saturan Recall y MRR. Un benchmark más grande, con preguntas ambiguas entre documentos y relevancia a nivel chunk, permitiría ajustar `WEIGHTS` y `TOP_K` con más precisión.
