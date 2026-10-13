# Capstone 2 — Multi-Agent RAG Customer Support

Enterprise customer-support automation using hybrid retrieval, a local Qwen2.5-1.5B-Instruct model, LangGraph multi-agent orchestration, FastAPI and Streamlit.

## Architecture

Knowledge base → cleaning/chunking → Sentence Transformers embeddings → FAISS semantic retrieval + BM25 keyword retrieval → hybrid context → query understanding → retrieval → sentiment → consultation → validation → retry/response.

## Features

- Document ingestion, cleaning and chunking with overlap
- FAISS semantic retrieval and BM25 keyword retrieval
- Hybrid retrieval
- Local Qwen LLM
- Multi-agent LangGraph workflow
- Reusable RAG and sentiment tools
- Conversation memory and feedback loop
- BLEU, ROUGE and retrieval relevance evaluation
- Baseline RAG vs Agentic RAG comparison
- FastAPI `/query`, `/health`, `/feedback`
- Streamlit chat UI
- GitHub Actions CI with Ruff and pytest

## Run locally

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open FastAPI Swagger at `http://localhost:8000/docs`.

In another terminal:

```bash
streamlit run ui/app.py
```

The local Qwen model is loaded on first query. A GPU is recommended.

## Docker

Docker is intentionally not included in this project.
