import os
from pathlib import Path
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv(".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

NAMESPACE = "ns1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

if not GROQ_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError("Faltan variables en el archivo .env (GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME).")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Simple chunker: divide texto en pedazos con solape."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += step
    return chunks


def load_txt_documents(folder: Path) -> list[tuple[str, str]]:
    """Devuelve lista de (filename, content) para todos los .txt en /data."""
    if not folder.exists():
        raise FileNotFoundError(f"No existe la carpeta: {folder}")

    files = sorted(folder.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No encontré archivos .txt en: {folder}")

    docs = []
    for fp in files:
        content = fp.read_text(encoding="utf-8", errors="ignore")
        docs.append((fp.name, content))
    return docs


def upsert_documents_from_data():
    docs = load_txt_documents(DATA_DIR)

    records = []
    for filename, content in docs:
        chunks = chunk_text(content, chunk_size=900, overlap=150)
        for j, ch in enumerate(chunks):
            records.append(
                {
                    "id": f"{filename}-{j}",
                    "text": ch,
                    "source": filename,
                    "chunk": j,
                }
            )

    if not records:
        raise ValueError("No se generaron chunks. Revisa que los .txt no estén vacíos.")

    
    index.upsert_records(NAMESPACE, records)
    print(f" Documentos subidos desde /data. Chunks subidos: {len(records)}. Namespace: {NAMESPACE}\n")


def retrieve_context(question: str, top_k: int = 3) -> str:
    response = index.search(
        namespace=NAMESPACE,
        query={
            "top_k": top_k,
            "inputs": {"text": question}
        }
    )

    hits = response.get("result", {}).get("hits", [])

    context_chunks = []
    for hit in hits:
        fields = hit.get("fields", {})
        txt = fields.get("text")
        if txt:
            context_chunks.append(txt)

    return "\n\n".join(context_chunks)


def generate_answer(question: str, context: str) -> str:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template(
        "Responde en español de forma clara y profesional.\n\n"
        "Regla: responde SOLO con base en el CONTEXTO. "
        "Si el contexto no alcanza, di: \"No tengo suficiente información en los documentos.\" \n\n"
        "CONTEXTO:\n{context}\n\n"
        "PREGUNTA:\n{question}\n\n"
        "RESPUESTA:"
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


if __name__ == "__main__":

    
    upsert_documents_from_data()

    
    questions = [
        "¿Qué información contiene el documento doc1.txt?",
        "Resume el contenido principal de doc2.txt en 2-3 líneas.",
        "¿Qué dice el documento sobre RAG?",
        "¿Qué papel cumple Pinecone en este sistema?"
    ]

    for question in questions:
        print("====================================")
        print("PREGUNTA:")
        print(question)

        context = retrieve_context(question, top_k=3)

        print("\n=== CONTEXTO RECUPERADO ===")
        print(context if context else "(No se recuperó contexto)")

        answer = generate_answer(question, context)

        print("\n=== RESPUESTA FINAL ===")
        print(answer)
        print("\n")