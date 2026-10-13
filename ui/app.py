"""Simple Streamlit chat UI."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise Customer Support", page_icon="💬")
st.title("Enterprise Customer Support")
st.caption("Multi-Agent RAG with LangGraph, hybrid retrieval and validation")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_session"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a customer-support question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "session_id": st.session_state.session_id},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        answer = data["answer"]
        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"Sources: {', '.join(data.get('sources', []))}")
            st.caption(f"Validated: {data.get('validation_passed', False)}")
    except requests.RequestException as exc:
        answer = f"API error: {exc}"
        with st.chat_message("assistant"):
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
