import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import ollama

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

TOP_K = 3
MAX_DISTANCE = 1.15
DECLINE_PHRASES = ("i don't know", "i do not know", "does not contain", "doesn't contain", "does not mention")

_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
_ollama_client = ollama.Client(host=OLLAMA_HOST)


def get_collection():
    """Fetch the existing Chroma collection created by ingest.py."""
    return _client.get_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Embed the question and return the top_k most similar chunks with their source metadata."""
    collection = get_collection()
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunks.append({"text": text, "source": meta["source"], "distance": dist})
    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Stuff retrieved chunks into a prompt so the model answers using only that context."""
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say you don't know — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, top_k: int = TOP_K) -> dict:
    """Full RAG flow: retrieve relevant chunks, then generate a grounded answer."""
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {"answer": "No documents found. Run ingestion first.", "sources": []}

    best = min((c["distance"] for c in chunks if "distance" in c), default=0)
    if best > MAX_DISTANCE:
        return {"answer": "I don't know. That doesn't seem to be covered in the documents.", "sources": []}

    prompt = build_prompt(question, chunks)

    response = _ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response["message"]["content"]
    declined = any(p in answer.lower().replace("’", "'") for p in DECLINE_PHRASES)
    sources = [] if declined else sorted(set(c["source"] for c in chunks))
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    result = generate_answer("What is retrieval-augmented generation?")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])