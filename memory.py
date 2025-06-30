# memory.py
import os, json, time, logging, pathlib

# DBs
from cachetools import LRUCache
from pymongo import MongoClient
import numpy as np

# RAG
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

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
chroma_client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=".chromadb"))
chroma_collection = chroma_client.get_or_create_collection("ltm")
sql_collection = chroma_client.get_or_create_collection("sql_chunks")
meta_collection = chroma_client.get_or_create_collection("table_meta")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def _embed(text: str) -> np.ndarray:
    return EMBED_MODEL.encode(text, normalize_embeddings=True)

import uuid
def add_ltm_entry(entry_type: str, question: str, sql: str = "", answer: str = "", rows: list = []):
    payload = {"question": question, "sql": sql, "rows": rows, "answer": answer}
    emb = EMBED_MODEL.encode([question])[0].tolist()
    collection = sql_collection if entry_type == "sql" else meta_collection
    collection.add(
        documents=[answer],
        metadatas=[payload],
        embeddings=[emb],
        ids=[str(uuid.uuid4())]
    )


def retrieve_ltm(question: str, top_k: int = 3) -> list[dict]:
    query_emb = EMBED_MODEL.encode([question])[0].tolist()
    # load all entries (for large scale, switch to MongoDB vector index)
    results = chroma_collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    # assumes documents hold answer/json payload
    return results["metadatas"][0]

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

def add_ltm_entry_dedup(question: str, sql: str, answer: str, rows: list):
    query_emb = EMBED_MODEL.encode([question])[0].tolist()
    result = sql_collection.query(query_embeddings=[query_emb], n_results=1, include=["metadatas"])

    # If existing similar SQL is found
    if result["metadatas"] and result["metadatas"][0]:
        existing = result["metadatas"][0][0]
        if sql.strip() == existing.get("sql", "").strip():
            # Optionally merge rows
            merged_rows = existing.get("rows", []) + rows
            merged_answer = answer or existing.get("answer", "")
            sql_collection.update(
                ids=[result["ids"][0][0]],
                metadatas=[{
                    "question": question,
                    "sql": sql,
                    "rows": merged_rows,
                    "answer": merged_answer,
                }]
            )
            return
    # Else add new
    sql_collection.add(
        documents=[answer],
        metadatas=[{"question": question, "sql": sql, "rows": rows, "answer": answer}],
        embeddings=[query_emb],
        ids=[str(uuid.uuid4())]
    )
