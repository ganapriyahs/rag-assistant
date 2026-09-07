from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_empty_question_returns_400():
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 400


@patch("app.main.generate_answer")
def test_query_returns_answer_and_sources(mock_generate_answer):
    mock_generate_answer.return_value = {
        "answer": "This is a mocked answer.",
        "sources": ["ml_basics.txt"],
    }

    response = client.post("/query", json={"question": "What is RAG?", "top_k": 3})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a mocked answer."
    assert data["sources"] == ["ml_basics.txt"]
    mock_generate_answer.assert_called_once_with("What is RAG?", top_k=3)


def test_query_missing_question_field_returns_422():
    response = client.post("/query", json={"top_k": 3})
    assert response.status_code == 422