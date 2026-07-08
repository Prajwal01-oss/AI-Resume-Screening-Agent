from sentence_transformers import SentenceTransformer

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def build_embedding(text: str):
    embedding = _MODEL.encode(text, convert_to_numpy=True)
    return {"text": text, "embedding": embedding}
