"""FastAPI service for the customer-support Agentic RAG system."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.memory import record_feedback, run_with_memory

app = FastAPI(
    title="Multi-Agent RAG Customer Support API",
    version="1.0.0",
    description="Enterprise customer-support automation using LangGraph, hybrid RAG, memory, validation, and feedback.",
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="Customer support question")
    session_id: str = Field(
        default="api_session", min_length=1, description="Conversation session identifier"
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    category: str | None = None
    sentiment: str | None = None
    urgency: str | None = None
    validation_passed: bool = False
    validation_reason: str | None = None
    agents_invoked: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)
    comment: str = ""


@app.get("/health", tags=["system"])
def health():
    """Return a lightweight service health response."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse, tags=["customer-support"])
def query(request: QueryRequest):
    """Run Agentic RAG with session-aware conversation memory."""
    result = run_with_memory(request.session_id, request.question)
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


@app.post("/feedback", tags=["customer-support"])
def feedback(request: FeedbackRequest):
    """Record explicit user feedback for a generated answer."""
    return record_feedback(**request.model_dump())
