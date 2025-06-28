import os, json, time, logging, pathlib
from cachetools import LRUCache
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np

log = logging.getLogger("memory-log")
log.info("🚀 Starting memory handler...")
# -------- STM: in-process LRU cache -----------
STM = LRUCache(maxsize=128)

def get_stm(question: str):
    return STM.get(question)

def add_stm(question: str, resp: dict):
    STM[question] = resp
    log.info("[STM] Cached response for `%s`", question)

# -------- LTM: MongoDB + embeddings -----------
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "cpg_ltm") # Fallback name
mongo = MongoClient(MONGO_URI)
db = mongo[DB_NAME]
ltm_coll = db["long_term_memory"]

# Vector embedding model loader
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def _embed(text: str) -> np.ndarray:
    return EMBED_MODEL.encode(text, normalize_embeddings=True)

def add_ltm_entry(question: str, sql: str, rows: list[dict], answer: str):
    emb = _embed(question).tolist()
    doc = {
        "question": question,
        "sql": sql,
        "rows": rows,
        "answer": answer,
        "embedding": emb,
        "ts": time.time()
    }
    ltm_coll.insert_one(doc)
    log.info("[LTM] Added entry `%s`", question)

def retrieve_ltm(question: str, top_k: int = 3) -> list[dict]:
    q_emb = np.array(_embed(question))
    # load all entries (for large scale, switch to MongoDB vector index)
    docs = list(ltm_coll.find({}, {"embedding":1, "question":1, "sql":1, "rows":1, "answer":1}))
    sims = []
    for d in docs:
        emb = np.array(d["embedding"])
        score = float(np.dot(q_emb, emb))
        sims.append((score, d))
    sims.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in sims[:top_k]]

## **Services**
def save(entry: dict):
    """
    Saves a memory entry to MongoDB LTM.
    Entry must contain a 'type' key, e.g. 'ddl', 'doc', or 'qa'.
    """
    if "type" not in entry:
        raise ValueError("Missing 'type' in memory entry")
    entry["ts"] = time.time()
    ltm_coll.insert_one(entry)
    log.info("[LTM] Saved entry of type `%s`", entry["type"])


def get_by_type(mem_type: str) -> list[dict]:
    return list(ltm_coll.find({"type": mem_type}))

def all() -> list[dict]:
    return list(ltm_coll.find())

def remove_by_hash(doc_id: str) -> bool:
    result = ltm_coll.delete_one({"_id": doc_id})
    return result.deleted_count > 0

def clear_all():
    STM.clear()
    ltm_coll.delete_many({})
    log.info("[MEMORY] All STM and LTM cleared.")