"""Knowledge-base ingestion, embeddings, FAISS and hybrid retrieval."""

import re
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, TOP_K

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    documents = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\s+", " ", text).strip()
        documents.append(Document(page_content=text, metadata={"source": path.name}))
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def build_retrieval_components():
    documents = load_documents()
    chunks = chunk_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )
    bm25 = BM25Okapi([tokenize(doc.page_content) for doc in chunks])
    return documents, chunks, embeddings, vectorstore, retriever, bm25


_components = None


def get_components():
    global _components
    if _components is None:
        _components = build_retrieval_components()
    return _components


def bm25_search(query: str, k: int = TOP_K):
    _, chunks, _, _, _, bm25 = get_components()
    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_indices]


def hybrid_search(query: str, k: int = TOP_K):
    _, _, _, _, retriever, _ = get_components()
    semantic_docs = retriever.invoke(query)
    keyword_docs = bm25_search(query, k=k)

    combined = []
    seen = set()
    for doc in semantic_docs + keyword_docs:
        key = (doc.metadata.get("source", ""), doc.page_content)
        if key not in seen:
            seen.add(key)
            combined.append(doc)
    return combined[:k]


def build_context(docs) -> str:
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', '')}\n{doc.page_content}" for doc in docs
    )
