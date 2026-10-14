"""Project configuration with an offline mode for environments without model access."""

import json
import os
import re
from functools import lru_cache

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "true").lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_generator():
    """Load Qwen only when online/full-model mode is explicitly enabled."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=160,
        temperature=0.0,
        do_sample=False,
        return_full_text=False,
    )


def _question_from_prompt(prompt: str) -> str:
    match = re.search(
        r"(?:Customer Question|Customer message|Question):\s*(.*?)(?:\n(?:Answer|Knowledge Base|$)|$)",
        prompt,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else prompt.strip()


def _category(question: str) -> str:
    text = question.lower()
    rules = {
        "refund": ("refund", "refunded", "money back", "returned"),
        "shipping": ("shipping", "delivery", "delivered", "shipment", "shipped"),
        "payment": ("payment", "upi", "debit", "credit card", "net banking"),
        "cancellation": ("cancel", "cancellation"),
        "account": ("password", "login", "account", "reset"),
    }
    for category, keywords in rules.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def _offline_llm(prompt: str) -> str:
    """Deterministic, KB-grounded fallback used when external model downloads are unavailable."""
    lower = prompt.lower()
    question = _question_from_prompt(prompt)

    if "valid json with intent, category, urgency" in lower:
        category = _category(question)
        urgency = "high" if any(x in question.lower() for x in ("urgent", "asap", "immediately")) else "medium"
        return json.dumps({
            "intent": "customer support request",
            "category": category,
            "urgency": urgency,
        })

    if "sentiment" in lower and "return only json" in lower:
        negative_words = ("angry", "bad", "terrible", "frustrated", "complaint", "not working", "problem")
        sentiment = "negative" if any(x in question.lower() for x in negative_words) else "neutral"
        urgency = "high" if any(x in question.lower() for x in ("urgent", "asap", "immediately")) else "medium"
        return json.dumps({"sentiment": sentiment, "urgency": urgency})

    if "determine whether the answer is fully supported" in lower:
        context_match = re.search(r"Knowledge Base:\s*(.*?)\s*Question:", prompt, re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r"Answer:\s*(.*?)\s*$", prompt, re.IGNORECASE | re.DOTALL)
        context = context_match.group(1) if context_match else ""
        answer = answer_match.group(1) if answer_match else ""
        context_words = set(re.findall(r"\b\w+\b", context.lower()))
        answer_words = set(re.findall(r"\b\w+\b", answer.lower()))
        overlap = len(context_words & answer_words) / max(len(answer_words), 1)
        supported = overlap >= 0.25
        return json.dumps({
            "supported": supported,
            "reason": "Answer terms are grounded in the retrieved knowledge base." if supported else "Answer is not sufficiently grounded in the retrieved knowledge base.",
        })

    if "knowledge base:" in lower and "customer question:" in lower:
        context_match = re.search(r"Knowledge Base:\s*(.*?)\s*Customer Question:", prompt, re.IGNORECASE | re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""
        if not context:
            return "I don't have enough information in the knowledge base to answer that."

        question_words = set(re.findall(r"\b\w+\b", question.lower()))
        blocks = re.split(r"\n\s*\n", context)
        best_text = ""
        best_score = 0
        for block in blocks:
            text = re.sub(r"^Source:\s*[^\n]+\n", "", block, flags=re.IGNORECASE).strip()
            words = set(re.findall(r"\b\w+\b", text.lower()))
            score = len(question_words & words)
            if score > best_score:
                best_score = score
                best_text = text
        return best_text if best_score else "I don't have enough information in the knowledge base to answer that."

    return "I don't have enough information in the knowledge base to answer that."


def local_llm(prompt: str, max_new_tokens: int = 200) -> str:
    if OFFLINE_MODE:
        return _offline_llm(prompt)
    generator = get_generator()
    response = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
    )
    return response[0]["generated_text"].strip()
