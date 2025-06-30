# memory.py
import os, logging, re, uuid, numpy as np

# DBs
from cachetools import LRUCache
from pymongo import MongoClient, ASCENDING
from sentence_transformers import SentenceTransformer

log = logging.getLogger("memory-log")
log.info("🚀 Starting memory handler...")

# Embedding model
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
def _embed(txt: str):   # returns list[float]
    return EMBED_MODEL.encode([txt], normalize_embeddings=True)[0].tolist()

# -------- STM: in-process LRU cache -----------
STM = LRUCache(maxsize=128)
get_stm  = STM.get
add_stm  = lambda q, r: STM.__setitem__(q, r)

# -------- LTM: MongoDB + embeddings -----------
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "cpg_ltm") # Fallback name
mongo = MongoClient(MONGO_URI)
db = mongo[DB_NAME]

# Whenever a new collection is created we add an index on norm_sql
def _get_collection(name: str):
    col = db[name]
    if "norm_sql_1" not in col.index_information():
        col.create_index([("norm_sql", ASCENDING)])
    return col

# ───────── Helpers
def _norm_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.lower()).strip()

# Lightweight regex fallback for table detection
_TABLE_RE = re.compile(r"\b(from|join)\s+([a-zA-Z0-9_]+)", re.I)
def _resolve_chunk(sql: str) -> str | None:
    m = _TABLE_RE.search(sql)
    return m.group(2).lower() if m else None

# ------------ PUBLIC API -----------------------
def add_sql_pair(
        question      : str,
        sql           : str,
        rows          : list,
        answer        : str,
        collection_id : str | None = None
    ) -> None:
    """
    Insert / merge a (question, SQL) pair into the right collection.

    • If `collection_id` is given, use it verbatim.
    • Otherwise resolve the first table name inside the SQL string.
    """
    coll_name = collection_id or _resolve_chunk(sql)
    if coll_name is None:
        coll_name = "unknown"
    # Get collection name
    col = _get_collection(coll_name)
    doc = {
        "question"  : question,
        "sql"       : sql,
        "norm_sql"  : _norm_sql(sql),
        "embedding" : _embed(question),
        "rows"      : rows[:2_000],
        "answer"    : answer
    }
    # Merge on identical norm_sql
    existing = col.find_one({"norm_sql": doc["norm_sql"]})
    if existing:
        merged = {**existing, **doc}
        col.update_one({"_id": existing["_id"]}, {"$set": merged})
        return
    col.insert_one(doc)

def retrieve_sql(
        query        : str,
        k            : int = 3,
        collection_id: str | list[str] | None = None
    ) -> list[dict]:
    """
    Semantic top-k retrieval from one or many collections.
    • If collection_id is None → search all collections lazily (slow but OK for <10k docs)
    • If list[str] → union of those
    • If str      → only that coll
    """
    target_colls: list[str]
    if collection_id is None:
        target_colls = [c for c in db.list_collection_names() if not c.startswith("system.")]
    elif isinstance(collection_id, str):
        target_colls = [collection_id]
    else:
        target_colls = collection_id
    # Embed query to compute sim-score
    query_emb = np.array(_embed(query))
    scored: list[tuple[float, dict]] = []
    # Retrieval and examination
    for name in target_colls:
        col = db[name]
        for d in col.find({}, {"embedding":1, "question":1, "sql":1, "rows":1, "answer":1}):
            emb = np.array(d["embedding"])
            scored.append((float(np.dot(query_emb, emb)), d))
    # Sort on sim-score
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]

# Save / get table context  (tiny docs per table)
def save_table_context(tbl, ctx): db["table_context"].update_one(
        {"tbl": tbl}, {"$set": {"context": ctx, "embedding": _embed(tbl)}}, upsert=True)
def get_table_context(tbl): doc = db["table_context"].find_one({"tbl": tbl}); return doc["context"] if doc else ""


# ------------ SERVICES --------------------------------------------------
# Maintenance
def clear_all():
    STM.clear()
    for n in db.list_collection_names():
        if not n.startswith("system."):
            db[n].delete_many({})
    log.info("🧹 Cleared STM and Mongo LTM")

