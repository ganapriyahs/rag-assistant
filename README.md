# 🔎 RAG Assistant

[![CI](https://github.com/ganapriyahs/rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/ganapriyahs/rag-assistant/actions/workflows/ci.yml)

A local, fully-functional Retrieval-Augmented Generation (RAG) system: documents are chunked, embedded, and stored in a vector database, then retrieved and used to ground answers from a locally-running LLM — no paid API required. Retrieval quality and "I don't know" behavior are measured with a small evaluation harness (see [Evaluation](#evaluation)).

[![RAG Assistant Demo](docs/demo.png)](docs/demo.png)

## What it does

Ask a question in plain English. The system:

1. Embeds the question using `sentence-transformers`
2. Retrieves the 3 most relevant chunks from a ChromaDB vector store, with their distances
3. If even the closest chunk is too far away (distance above 1.15), answers "I don't know" without calling the model
4. Otherwise passes the chunks as context to a local LLM (via Ollama), instructed to answer only from that context
5. Returns the answer along with its source document(s); sources are omitted when the model declines to answer

This reduces hallucination by constraining the model to retrieved context and refusing off-topic questions, and it keeps a link between answers and sources. How well this works is measured, not assumed; see [Evaluation](#evaluation) for results and limitations.

## Architecture

```
data/sample_docs/ → chunking (500 chars, 50 overlap) → embeddings → ChromaDB (vector store)
                                                                          ↓
user question → embed → similarity search (top 3, with distances)
                                ↓
              closest chunk too far? → "I don't know" (no model call)
                                ↓
        retrieved chunks + question → Ollama (local LLM)
                                ↓
              FastAPI (/query) → Streamlit UI
```

## Project structure

```
app/            ingestion, retrieval + generation, FastAPI app
ui/             Streamlit front end
tests/          pytest unit tests (LLM calls mocked)
eval/           retrieval and answer-behavior evaluation scripts + question set
data/sample_docs/   documents to ingest (.txt)
docs/           demo screenshot
```

## Tech stack

- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2, runs locally, no API cost)
- **Vector store:** ChromaDB (persistent, local)
- **LLM:** Ollama (llama3.2, runs fully locally — no API key needed)
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Testing:** pytest, with mocked LLM calls for fast, CI-safe tests
- **CI/CD:** GitHub Actions (runs test suite on every push)
- **Containerization:** Docker + docker-compose

## Running locally

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed

```bash
# 1. Clone and set up environment
git clone https://github.com/ganapriyahs/rag-assistant.git
cd rag-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Pull the local LLM
ollama pull llama3.2
ollama serve   # keep running in a separate terminal

# 3. Configure environment
cp .env.example .env

# 4. Ingest documents (add your own .txt files to data/sample_docs/ first)
python -m app.ingest

# 5. Run the API (separate terminal)
python -m uvicorn app.main:app --reload

# 6. Run the UI (separate terminal)
streamlit run ui/streamlit_app.py
```

Visit `http://localhost:8501` for the UI, or `http://localhost:8000/docs` for the interactive API docs.

**PDFs:** ingestion reads `.txt` files only. Convert PDFs with `pdftotext` (from poppler, e.g. `brew install poppler`) rather than `pypdf`, which splits words containing the "fi" and "fl" ligatures:

```bash
pdftotext data/sample_docs/your_document.pdf data/sample_docs/your_document.txt
```

ChromaDB prints harmless telemetry warnings (`Failed to send telemetry event ...`); they do not affect results.

## Running with Docker

```bash
docker compose up --build
```

Requires `ollama serve` running on the host machine — the container reaches it via `host.docker.internal`.

## Running tests

```bash
python -m pytest -v
```

Tests mock LLM calls so they run instantly and don't require Ollama — this is also what makes CI possible without a self-hosted runner. These tests check the API and the chunking function; they do not measure retrieval quality, which is what the evaluation below is for.

## Evaluation

I tested the system against *Inference Engineering* by Philip Kiely (Baseten Books). The book is not included in this repo. To reproduce, obtain the PDF, extract it with `pdftotext` (see above), then:

```bash
python -m app.ingest
python -m eval.run_eval        # retrieval: hit@1 / hit@3
python -m eval.run_answers     # answer behavior on unanswerable and control questions (needs Ollama)
python -m eval.distances       # distance of the closest chunk for every question
python eval/compare.py a.json b.json   # which questions changed rank between two runs
```

**Method.** `eval/questions.csv` has 16 answerable questions, each with one or more short key phrases (separated by `|`) taken from the passage that answers it, plus 3 unanswerable questions. A question counts as a hit@k if any key phrase appears in any of the top k retrieved chunks, compared with whitespace removed and case ignored. Result files are not committed because they contain excerpts of the book.

**Retrieval results** (16 answerable questions):

| Setup | hit@1 | hit@3 |
|---|---|---|
| 500/50 chunks, pypdf text, strict keys | 11/16 | 14/16 |
| Same, 2 keys widened after review | 12/16 | 16/16 |
| 1000/200 chunks | 12/16 | 15/16 |
| 500/50 chunks, pdftotext (cleaned) text | 10/16 | 15/16 |

Two keys were widened after I found the book explains the same idea in more than one place. The rows are not all directly comparable: switching to cleaned text changed every chunk boundary. Differences of one or two questions are within noise on a set this small. The 1000/200 setting was not clearly better than 500/50.

**Unanswerable questions.** All 3 were declined. Two (the capital of France, training a sentiment classifier) are stopped by the distance cutoff before the model runs. The third (PyTorch Lightning) is near the book's topic, so retrieval cannot reject it by distance; the model declined it, and the sources are then hidden.

**Limitations.**
- Only 16 answerable questions, written by me after reading the passages, so they echo the book's wording and are easier than real user questions.
- A key phrase marks one passage, and the book often explains an idea in several places, so some misses are false misses.
- The distance cutoff (1.15) was tuned on 19 questions and sits close to the farthest answerable one (1.05). It should be re-checked as the question set grows.
- Decline detection matches phrases in the model's reply, so a differently worded decline would still show sources.
- Only one document has been evaluated, and the unanswerable questions were all easy to reject.

## What this project demonstrates

- End-to-end RAG pipeline design (chunking strategy, embedding, retrieval, grounded generation)
- API design with FastAPI, including request/response validation via Pydantic
- Testing strategy for systems with external/local dependencies (mocking the LLM layer)
- Measuring retrieval quality instead of assuming it: hit@k, before/after experiments, and recorded limitations
- Handling "I don't know": a distance guard, decline detection, and source suppression
- Containerization and cross-container networking (bridging Docker to a host-level service)
- CI/CD setup with GitHub Actions
- Clean separation of concerns: ingestion, retrieval logic, API layer, and UI are fully decoupled

## Possible extensions

- Expand the evaluation to about 40 questions with several acceptable passages each, and add harder "near the topic but not in the document" questions
- Paragraph- or sentence-aware chunking to avoid mid-sentence cuts
- Add built-in PDF/DOCX ingestion instead of the manual `pdftotext` step
- Add a re-ranking step after initial retrieval for better relevance
- Swap Ollama for a hosted LLM via environment variable (already scaffolded — the client just needs a different base URL)
- Persist chat history in the Streamlit UI