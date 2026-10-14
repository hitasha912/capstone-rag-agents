# Deployment Guide

## FastAPI

Install dependencies and start the API:

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open `/docs` for Swagger UI and `/health` for the health check.

## Streamlit

In a second terminal:

```bash
streamlit run ui/app.py
```

Set `API_URL` if the API is hosted remotely.

## Suggested deployment

Use a Python-compatible cloud service that can run a long-lived FastAPI process. Configure the service to install from `requirements.txt` and use:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

The application currently uses the local Qwen model and Hugging Face embedding model, so the deployment environment needs enough memory/storage for model loading. No Docker is required for this project.

## Demo API requests

Health:

```bash
curl /health
```

Query:

```bash
curl -X POST /query -H "Content-Type: application/json" -d '{"session_id":"demo","question":"How long does a refund take?"}'
```

Feedback:

```bash
curl -X POST /feedback -H "Content-Type: application/json" -d '{"session_id":"demo","question":"How long does a refund take?","answer":"Refunds are processed within 5-7 business days after inspection and approval.","rating":5,"comment":"Helpful"}'
```
