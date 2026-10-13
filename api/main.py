"""FastAPI service for the customer-support Agentic RAG system."""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.graph import run_agentic_rag
from src.memory import record_feedback

app = FastAPI(title="Multi-Agent RAG Customer Support API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = Field(default="api_session", min_length=1)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []
    category: Optional[str] = None
    sentiment: Optional[str] = None
    urgency: Optional[str] = None
    validation_passed: bool = False
    validation_reason: Optional[str] = None
    agents_invoked: list[str] = []


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    rating: int = Field(ge=1, le=5)
    comment: str = ""


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = run_agentic_rag(request.question, request.session_id)
    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        category=result.get("category"),
        sentiment=result.get("sentiment"),
        urgency=result.get("urgency"),
        validation_passed=result.get("validation_passed", False),
        validation_reason=result.get("validation_reason"),
        agents_invoked=result.get("agents_invoked", []),
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    return record_feedback(**request.model_dump())
