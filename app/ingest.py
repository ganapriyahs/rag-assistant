import os
import re
from bisect import bisect_right
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

# Running page headers look like "Chapter 3: Hardware". Table-of-contents lines
# have dotted leaders ("Chapter 3: Hardware.....71"), so the title must contain no dots.
CHAPTER_HEADER = re.compile(r"^Chapter (\d+): ([^.]+?)\s*$")
# The PDF has 5 pages of front matter before printed page 1.
PAGE_OFFSET = 0


def chunk_with_offsets(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[tuple[int, str]]:
    """Split text into overlapping chunks, returning (start_offset, chunk) pairs."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append((start, text[start:end]))
        start += chunk_size - overlap
    return [(s, c.strip()) for s, c in chunks if c.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks so retrieval can find focused, relevant passages."""
    return [c for _, c in chunk_with_offsets(text, chunk_size, overlap)]


def build_page_index(text: str) -> list[tuple[int, int, str]]:
    """Return one (start_offset, printed_page, chapter) entry per page.

    Pages are separated by form feed characters. Text without form feeds is
    treated as a single unpaged document: page 0, no chapter.
    """
    if "\f" not in text:
        return [(0, 0, "")]

    index = []
    chapter = ""
    offset = 0
    for pdf_page, page_text in enumerate(text.split("\f"), start=1):
        for line in page_text.splitlines():
            m = CHAPTER_HEADER.match(line.strip())
            if m:
                chapter = f"Chapter {m.group(1)}: {m.group(2)}"
        index.append((offset, pdf_page - PAGE_OFFSET, chapter))
        offset += len(page_text) + 1  # +1 for the form feed itself
    return index


def locate(index: list[tuple[int, int, str]], offset: int) -> tuple[int, str]:
    """Find the (page, chapter) that a character offset falls in."""
    starts = [s for s, _, _ in index]
    k = bisect_right(starts, offset) - 1
    _, page, chapter = index[k]
    return page, chapter


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
        page_index = build_page_index(doc["text"])
        for i, (offset, chunk) in enumerate(chunk_with_offsets(doc["text"])):
            page, chapter = locate(page_index, offset)
            ids.append(f"{doc['source']}_{i}")
            texts.append(chunk)
            metadatas.append(
                {
                    "source": doc["source"],
                    "chunk_index": i,
                    "page": page,
                    "chapter": chapter,
                }
            )

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"Ingested {len(texts)} chunks from {len(docs)} document(s) into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_vector_store()