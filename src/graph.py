"""LangGraph orchestration for the multi-agent RAG workflow."""

from langgraph.graph import END, START, StateGraph

from .agents import (
    AgentState,
    consultation_agent,
    query_understanding_agent,
    retrieval_agent,
    sentiment_agent,
    validation_agent,
)


def should_retry(state: AgentState):
    if state.get("validation_passed", False):
        return "end"
    return "retry" if state.get("retry_count", 0) < 2 else "end"


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("query_understanding", query_understanding_agent)
    workflow.add_node("retrieval", retrieval_agent)
    workflow.add_node("sentiment", sentiment_agent)
    workflow.add_node("consultation", consultation_agent)
    workflow.add_node("validation", validation_agent)
    workflow.add_edge(START, "query_understanding")
    workflow.add_edge("query_understanding", "retrieval")
    workflow.add_edge("retrieval", "sentiment")
    workflow.add_edge("sentiment", "consultation")
    workflow.add_edge("consultation", "validation")
    workflow.add_conditional_edges("validation", should_retry, {"retry": "retrieval", "end": END})
    return workflow.compile()


graph = build_graph()


def run_agentic_rag(question: str, session_id: str = "session_001"):
    return graph.invoke({
        "session_id": session_id,
        "question": question,
        "retry_count": 0,
        "agents_invoked": [],
        "feedback": None,
    })
