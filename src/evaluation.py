"""BLEU, ROUGE and retrieval relevance evaluation utilities."""

import re

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

from .rag import get_components, hybrid_search


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def evaluate_answer(reference: str, prediction: str):
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([tokenize(reference)], tokenize(prediction), smoothing_function=smooth)
    scores = rouge.score(reference, prediction)
    return {
        "BLEU": bleu,
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure,
    }


def retrieval_relevance(query: str, docs):
    _, _, embeddings, _, _, _ = get_components()
    if not docs:
        return 0.0
    query_vector = embeddings.embed_query(query)
    doc_vectors = [embeddings.embed_query(doc.page_content) for doc in docs]
    return float(np.mean(cosine_similarity([query_vector], doc_vectors)[0]))


EVALUATION_DATA = [
    {"question": "How long does a refund take?", "reference": "Refunds are processed within 5-7 business days after the returned product is inspected and approved."},
    {"question": "How long does standard shipping take?", "reference": "Standard shipping usually takes 5-7 business days."},
    {"question": "Can I cancel an order after it has shipped?", "reference": "No. Orders cannot be cancelled once they have been shipped."},
    {"question": "What payment methods are accepted?", "reference": "Credit cards, debit cards, UPI and net banking are accepted."},
    {"question": "How can I reset my password?", "reference": "Customers can reset their password using the Forgot Password option."},
]
