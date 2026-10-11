import hashlib
import logging
import math
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logger = logging.getLogger("uvicorn.error")


def _fallback_embedding(text: str, dim: int = 512) -> list[float]:
    """Deterministic hashing vectorizer used when OpenAI credits are exhausted."""
    tokens = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
    vec = [0.0] * dim
    for token in tokens:
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0
    # Also include bigrams for context
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]}_{tokens[i+1]}"
        idx = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.5

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def build_embedding(text: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            return {"text": text, "embedding": embedding}
        except Exception as exc:
            logger.warning(f"OpenAI embedding call failed ({exc}). Using fallback vectorizer.")

    # Graceful fallback so the application never breaks
    embedding = _fallback_embedding(text)
    return {"text": text, "embedding": embedding}

