from fastapi.testclient import TestClient

from api.main import app
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
