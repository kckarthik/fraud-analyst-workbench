"""
Thin wrapper over the local Ollama server. Deliberately plain httpx rather
than a LangChain model wrapper — LangGraph nodes are just functions over
state, they don't need LangChain's model abstractions, and skipping that
layer keeps the dependency footprint (and RAM footprint) smaller.
"""
import os

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("AGENT_MODEL", "llama3.2:3b")


def generate(prompt: str, system: str | None = None, temperature: float = 0.1) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = httpx.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json()["response"].strip()
