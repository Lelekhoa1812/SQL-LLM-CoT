# memory.py
import os, json, time, logging, pathlib, re

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

# Vector embedding model loader and chroma
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chromadb")
chroma_client = Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_DIR,
        anonymized_telemetry=False
    )
)
chroma_collection = chroma_client.get_or_create_collection("ltm")
SQL_COL  = chroma_client.get_or_create_collection("sql_pairs")
META_COL = chroma_client.get_or_create_collection("table_meta")

def _uuid():            # tiny helper
    import uuid; return str(uuid.uuid4())

def _embed(txt: str):   # returns list[float]
    return EMBED_MODEL.encode([txt], normalize_embeddings=True)[0].tolist()

# ------------ PUBLIC API -------------------------------------------------

def add_sql_pair(question: str, sql: str, rows: list, answer: str):
    """
    • Dedup on *SQL text* (case/space-insensitive)  
    • If already present -> merge rows & keep best answer length>0
    """
    norm = re.sub(r"\s+", " ", sql.lower().strip())
    existing = SQL_COL.query(
        query_texts=[norm], n_results=1, include=["documents", "metadatas"]
    )
    if existing["metadatas"] and existing["metadatas"][0]:
        doc_id  = existing["ids"][0][0]
        meta    = existing["metadatas"][0][0]
        merged  = {
            "question": question,
            "sql": sql,
            "rows": (meta["rows"] + rows)[:2_000],   # cap
            "answer": answer or meta.get("answer", "")
        }
        SQL_COL.update(ids=[doc_id], metadatas=[merged])
        return

    SQL_COL.add(
        ids=[_uuid()],
        documents=[answer],
        embeddings=[_embed(question)],
        metadatas=[{
            "question": question, "sql": sql,
            "rows": rows, "answer": answer
        }]
    )

def add_table_meta(tbl_name: str, doc: str):
    META_COL.add(
        ids=[_uuid()],
        documents=[doc],
        embeddings=[_embed(tbl_name)],
        metadatas=[{"table": tbl_name}]
    )

def retrieve_sql(question: str, k: int = 3) -> list[dict]:
    res = SQL_COL.query(
        query_embeddings=[_embed(question)],
        n_results=k,
        include=["metadatas"]
    )
    return res["metadatas"][0] if res["metadatas"] else []


# ------------ SERVICES --------------------------------------------------
def clear_all():
    STM.clear()
    SQL_COL.delete(where={})
    META_COL.delete(where={})
    log.info("[MEMORY] All STM and LTM cleared 🧹.")

