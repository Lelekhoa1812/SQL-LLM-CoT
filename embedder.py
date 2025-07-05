import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> list[float]:
    return EMBED_MODEL.encode([text], normalize_embeddings=True)[0].tolist()

def embed_np(text: str) -> np.ndarray:
    return EMBED_MODEL.encode(text, normalize_embeddings=True)
