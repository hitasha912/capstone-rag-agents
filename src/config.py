"""Project configuration and lazy local LLM loading."""

import os
from functools import lru_cache

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4


@lru_cache(maxsize=1)
def get_generator():
    """Load the local Qwen generator only when it is first needed."""
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


def local_llm(prompt: str, max_new_tokens: int = 200) -> str:
    generator = get_generator()
    response = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
    )
    return response[0]["generated_text"].strip()
