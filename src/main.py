import asyncio

from src.ingest import ingest
from src.rag import get_rag_response


async def main():
    ingest()

    while True:
        pregunta = input("\nPregunta (enter para salir): ").strip()
        if not pregunta:
            break
        resultado = await get_rag_response(pregunta)
        print(f"Respuesta: {resultado.respuesta}")
        print(f"Referencias: {resultado.referencias}")


if __name__ == "__main__":
    asyncio.run(main())
