# embedder.py
from typing import List
import requests

class OllamaEmbedder:
    """
    Simple Ollama embeddings client.
    Requires: ollama pull nomic-embed-text
    Ollama must be running (http://localhost:11434)
    """
    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.host = host.rstrip("/")
        self.model = model

    def embed(self, text: str) -> List[float]:
        url = f"{self.host}/api/embeddings"
        r = requests.post(url, json={"model": self.model, "prompt": text}, timeout=60)
        r.raise_for_status()
        return r.json()["embedding"]
