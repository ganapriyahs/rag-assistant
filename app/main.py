from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_chain import generate_answer

app = FastAPI(title="RAG Assistant API")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    locations: list[str] = []


@app.get("/health")
def health():
    """Basic liveness check — confirms the API is up and reachable."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Accepts a question, runs the full RAG pipeline, returns a grounded answer with sources."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = generate_answer(request.question, top_k=request.top_k)
    return QueryResponse(**result)