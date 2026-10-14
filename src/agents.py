"""Agents and reusable tools for the LangGraph customer-support workflow."""

import json
import re
from typing import Any, TypedDict

from langchain_core.tools import tool

from .config import local_llm
from .rag import build_context, hybrid_search


class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    intent: str
    category: str
    urgency: str
    retrieved_context: str
    sources: list[str]
    sentiment: str
    answer: str
    confidence: float
    validation_passed: bool
    validation_reason: str
    retry_count: int
    agents_invoked: list[str]
    feedback: dict[str, Any] | None


def _json_from(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def planner_agent(state: AgentState):
    """Create a small explicit execution plan before specialist agents run."""
    return {
        "intent": "customer support request",
        "agents_invoked": state.get("agents_invoked", []) + ["planner"],
    }


def query_understanding_agent(state: AgentState):
    raw = local_llm(
        f"""Analyze this customer support query. Return ONLY valid JSON with intent, category, urgency.
Category: refund, shipping, payment, cancellation, account, other.
Urgency: low, medium, high.
Question: {state['question']}""",
        max_new_tokens=120,
    )
    data = _json_from(raw)
    return {
        "intent": data.get("intent", state.get("intent", "customer support request")),
        "category": data.get("category", "other"),
        "urgency": data.get("urgency", "medium"),
        "agents_invoked": state.get("agents_invoked", []) + ["query_understanding"],
    }


def retrieval_agent(state: AgentState):
    """Retrieve grounded context by explicitly invoking the reusable RAG tool."""
    context = rag_search.invoke({"question": state["question"]})
    source_matches = re.findall(r"Source:\s*([^\n]+)", context)
    return {
        "retrieved_context": context,
        "sources": list(dict.fromkeys(source_matches)),
        "retry_count": state.get("retry_count", 0) + 1,
        "agents_invoked": state.get("agents_invoked", []) + ["retrieval", "rag_search_tool"],
    }


def sentiment_agent(state: AgentState):
    """Analyze sentiment by explicitly invoking the reusable sentiment tool."""
    raw = sentiment_analysis_tool.invoke({"question": state["question"]})
    data = _json_from(raw)
    return {
        "sentiment": data.get("sentiment", "neutral"),
        "urgency": data.get("urgency", state.get("urgency", "medium")),
        "agents_invoked": state.get("agents_invoked", []) + ["sentiment", "sentiment_analysis_tool"],
    }


def consultation_agent(state: AgentState):
    prompt = f"""You are an enterprise customer support response agent.
Use ONLY the knowledge-base context. Never invent information or contradict policy. Preserve exact numbers and time periods. If unsupported, say: I don't have enough information in the knowledge base to answer that.
Knowledge Base:
{state['retrieved_context']}
Customer Question:
{state['question']}
Answer:"""
    return {
        "answer": local_llm(prompt, max_new_tokens=180),
        "agents_invoked": state.get("agents_invoked", []) + ["consultation"],
    }


def validation_agent(state: AgentState):
    raw = local_llm(
        f"""Determine whether the answer is fully supported by the knowledge base. Return ONLY JSON: {{"supported": true, "reason": "brief reason"}}
Knowledge Base: {state['retrieved_context']}
Question: {state['question']}
Answer: {state['answer']}""",
        max_new_tokens=120,
    )
    data = _json_from(raw)
    supported = data.get("supported")
    if isinstance(supported, str):
        supported = supported.lower() == "true"
    if supported is None:
        context_words = set(re.findall(r"\b\w+\b", state["retrieved_context"].lower()))
        answer_words = set(re.findall(r"\b\w+\b", state["answer"].lower()))
        overlap = len(context_words & answer_words) / max(len(answer_words), 1)
        supported = overlap >= 0.25
    return {
        "validation_passed": bool(supported),
        "validation_reason": data.get("reason", "Validation completed."),
        "confidence": 1.0 if supported else 0.0,
        "agents_invoked": state.get("agents_invoked", []) + ["validation"],
    }


@tool
def rag_search(question: str) -> str:
    """Search the customer-support knowledge base and return relevant context."""
    return build_context(hybrid_search(question, k=4))


@tool
def sentiment_analysis_tool(question: str) -> str:
    """Analyze customer sentiment and urgency."""
    return local_llm(
        f'Return ONLY JSON: {{"sentiment":"positive|neutral|negative","urgency":"low|medium|high"}}\nCustomer message: {question}',
        max_new_tokens=100,
    )
