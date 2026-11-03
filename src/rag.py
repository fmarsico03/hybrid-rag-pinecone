import asyncio
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import GOOGLE_API_KEY, LLM_MODEL
from src.retriever import RAGSystem


class RAGResponse(BaseModel):
    respuesta: str = Field(description="Respuesta basada solo en el contexto, o 'No lo sé'")
    referencias: list[str] = Field(
        description="Fuentes usadas, con el formato 'archivo.pdf p.N'. Vacía si la respuesta es 'No lo sé'"
    )


SYSTEM_PROMPT = """Eres un asistente técnico. Responde SOLO basándote en el CONTEXTO proporcionado.
Si la respuesta no está en el CONTEXTO, responde exactamente "No lo sé" y deja las referencias vacías.
No uses conocimiento propio ni inventes información. Responde en el idioma de la pregunta.
En referencias, incluye únicamente las fuentes del CONTEXTO que usaste.

{format_instructions}"""

HUMAN_PROMPT = """CONTEXTO:
{context}

PREGUNTA: {question}"""


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[fuente: {doc.metadata['source']} p.{doc.metadata['page']}]\n{doc.page_content}"
        for doc in docs
    )


@lru_cache(maxsize=1)
def get_chain() -> Runnable:
    """Arma la cadena una sola vez y recién cuando se usa (el retriever necesita la ingesta hecha)."""
    parser = PydanticOutputParser(pydantic_object=RAGResponse)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]).partial(format_instructions=parser.get_format_instructions())

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, google_api_key=GOOGLE_API_KEY)

    return (
        {"context": RunnableLambda(RAGSystem().retrieve) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )


async def get_rag_response(query: str) -> RAGResponse:
    return await get_chain().ainvoke(query)


async def main():
    pruebas = [
        # Pregunta cuya respuesta está en los documentos
        "What Python version does ADK require?",
        # Pregunta trampa: no está en los documentos
        "What is the capital of Australia?",
    ]
    for pregunta in pruebas:
        respuesta = await get_rag_response(pregunta)
        print(f"\nPregunta: {pregunta}")
        print(f"Respuesta: {respuesta.respuesta}")
        print(f"Referencias: {respuesta.referencias}")


if __name__ == "__main__":
    asyncio.run(main())
