import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")
DATA_DIR = Path(__file__).parent.parent / "data" / "sample_docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks so retrieval can find focused, relevant passages."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Read every .txt file in data_dir and return list of {source, text} dicts."""
    docs = []
    for filepath in data_dir.glob("*.txt"):
        text = filepath.read_text(encoding="utf-8")
        docs.append({"source": filepath.name, "text": text})
    return docs


def build_vector_store():
    """Load docs, chunk them, embed them, and persist into ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Recreate collection fresh each time we re-ingest
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    docs = load_documents()
    if not docs:
        print(f"No .txt files found in {DATA_DIR}. Add some and re-run.")
        return

    ids, texts, metadatas = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc['source']}_{i}")
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk_index": i})

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"Ingested {len(texts)} chunks from {len(docs)} document(s) into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_vector_store()