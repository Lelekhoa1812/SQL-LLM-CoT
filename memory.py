# memory.py
import os, logging, re, uuid, json, numpy as np
from datetime import datetime, timedelta

# DBs
from cachetools import LRUCache
from pymongo import MongoClient, ASCENDING

# Embed
import os
os.environ["HF_HOME"] = "/tmp/hf_cache"  # write-safe
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/hf_cache/sentence-transformers"
from embedder import _embed

log = logging.getLogger("memory-log")
log.info("🚀 Starting memory handler...")

# -------- STM: in-process LRU cache -----------
STM = LRUCache(maxsize=128)
get_stm = lambda q: STM.get(_norm_question(q))
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
    if "expire_at_1" not in col.index_information():
        col.create_index([("expire_at", ASCENDING)], expireAfterSeconds=60 * 60 * 24 * 90)  # 90d
    return col

# ───────── Helpers
def _norm_sql(sql: str) -> str:
    '''
    Frequent filter condition, and without an index, MongoDB will do a collection scan every time. With a proper index:
    + Lookup time drops from O(n) → O(log n)
    + Especially critical once each table's collection reaches thousands of rows.
    '''
    return re.sub(r"\s+", " ", sql.lower()).strip()
def _norm_question(q: str) -> str:
    return re.sub(r"[^\w\s]", "", q.lower()).strip()

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
    # Merge on identical norm_sql
    norm = _norm_sql(sql)
    existing = col.find_one({"norm_sql": norm})
    if existing:
        log.info(f"[LTM] Skipped duplicate SQL in `{coll_name}`: {sql[:80]}")
        merged = {**existing, **doc}
        col.update_one({"_id": existing["_id"]}, {"$set": merged})
        return
    # Insert new doc
    doc = {
        "question"  : question,
        "sql"       : sql,
        "norm_sql"  : norm,
        "embedding" : _embed(question),
        "rows"      : rows[:2_000],
        "answer"    : answer,
        "expire_at" : datetime.utcnow() + timedelta(days=90)
    }
    col.insert_one(doc)
    log.info(f"[LTM] ✅ Inserted new pair into `{coll_name}`: {question[:60]}")

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
            if "question" not in d or "sql" not in d:
                continue # skip malformed docs
            emb = np.array(d["embedding"])
            scored.append((float(np.dot(query_emb, emb)), d))
    # Sort on sim-score
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]

# Save / get table context (tiny docs per table)
def save_table_context(tbl, ctx):
    """
    Save per-table summary context (as dict or JSON string) and embed table name + context for RAG.
    """
    if isinstance(ctx, dict): ctx_str = json.dumps(ctx)
    else: ctx_str = ctx  # assume already a JSON string
    embedding_source = ctx_str 
    db["table_context"].update_one(
        {"tbl": tbl},
        {"$set": {
            "context": ctx_str,
            "embedding": _embed(embedding_source)
        }},
        upsert=True
    )
def get_table_context(tbl: str) -> dict:
    doc = db["table_context"].find_one({"tbl": tbl})
    if not doc or "context" not in doc:
        return {}
    try:
        parsed = json.loads(doc["context"])
        logging.info(f"[LTM] Loaded table context for `{tbl}`: {parsed}")
        return parsed
    except Exception as e:
        db["table_context"].delete_one({"tbl": tbl})
        log.warning(f"[memory] Invalid context for {tbl}, purged. Error: {e}")
        return {}

# ------------ SERVICES --------------------------------------------------
# Maintenance
def clear_all():
    STM.clear()
    for n in db.list_collection_names():
        if not n.startswith("system."):
            db[n].delete_many({})
    log.info("🧹 Cleared STM and Mongo LTM")

def start_up_create_indexes():
    for tbl in db.list_collection_names():
        db[tbl].create_index([("norm_sql", ASCENDING)])

def count_pairs(tbl:str)->int:
    col = _get_collection(tbl)
    return col.estimated_document_count()
