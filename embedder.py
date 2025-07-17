import os, numpy as np
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
hf_token = os.getenv("HF_TOKEN")
EMBED_MODEL = SentenceTransformer(model_name, token=hf_token)
                                  
def _embed(text: str) -> list[float]:
    return EMBED_MODEL.encode([text], normalize_embeddings=True)[0].tolist()

def _embed_np(text: str) -> np.ndarray:
    return EMBED_MODEL.encode(text, normalize_embeddings=True)
