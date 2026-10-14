from fastapi.testclient import TestClient

from api.main import app
from src.agents import rag_search, sentiment_analysis_tool
from src.memory import record_feedback

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_feedback_validation():
    row = record_feedback("test_session", "test question", "test answer", 5, "good")
    assert row["rating"] == 5


def test_feedback_rejects_invalid_rating():
    response = client.post(
        "/feedback",
        json={
            "session_id": "test",
            "question": "q",
            "answer": "a",
            "rating": 6,
            "comment": "bad",
        },
    )
    assert response.status_code == 422


def test_agents_as_tools_are_registered():
    assert rag_search.name == "rag_search"
    assert sentiment_analysis_tool.name == "sentiment_analysis_tool"
    assert "knowledge base" in rag_search.description.lower()
    assert "sentiment" in sentiment_analysis_tool.description.lower()
