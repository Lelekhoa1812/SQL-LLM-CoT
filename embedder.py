from sentence_transformers import SentenceTransformer

# Embedding model
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
def _embed(txt: str):   # returns list[float]
    return EMBED_MODEL.encode([txt], normalize_embeddings=True)[0].tolist()
