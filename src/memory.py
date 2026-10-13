"""Conversation memory and user feedback storage."""

import csv
from datetime import datetime
from pathlib import Path

from .graph import run_agentic_rag

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = ROOT / "feedback.csv"
conversation_memory = {}


def run_with_memory(session_id: str, question: str):
    history = conversation_memory.get(session_id, [])
    contextual_question = question
    if history:
        history_text = "\n".join(
            f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in history[-4:]
        )
        contextual_question = f"""Conversation history:\n{history_text}\n\nCurrent user question:\n{question}\n\nUse the conversation history only to understand references. Use the knowledge base for factual answers."""
    result = run_agentic_rag(contextual_question, session_id=session_id)
    conversation_memory.setdefault(session_id, []).append({"question": question, "answer": result["answer"]})
    return result


def record_feedback(session_id: str, question: str, answer: str, rating: int, comment: str = ""):
    rating = int(rating)
    if not 1 <= rating <= 5:
        raise ValueError("rating must be between 1 and 5")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment,
    }
    file_exists = FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return row
