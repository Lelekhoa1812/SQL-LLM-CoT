# utils.py
import functools, logging, os, pandas as pd
from sqlalchemy import create_engine, inspect, text
from databases import Database

log = logging.getLogger("utils-log")
log.info("🚀 Starting memory utils...")

# Prefixes
user = os.getenv("DB_USER")
pw = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT", "3306")
db = os.getenv("DB_NAME")

# ---------- MySQL Engine ----------
DB_CFG = {
    "user":     user,
    "password": pw,
    "host":     host,
    "port":     "3306",
    "db":       db,
}
try:
    ENGINE = create_engine(
        f"mysql+pymysql://{DB_CFG['user']}:{DB_CFG['password']}@"
        f"{DB_CFG['host']}:{DB_CFG['port']}/{DB_CFG['db']}",
        pool_recycle=3600,
    )
except Exception as e:
    log.error("🚫 Could not connect to MySQL: %s", e)
    ENGINE = None

DATABASE = Database(f"mysql+pymysql://{DB_CFG['user']}:{DB_CFG['password']}@"
                    f"{DB_CFG['host']}:{DB_CFG['port']}/{DB_CFG['db']}")


# ---------- Logging ----------
log = logging.getLogger("utils-service")
log.info("🚀 Starting utils...")

# ---------- Simple in-process LRU cache ----------
@functools.lru_cache(maxsize=32)
def _cached_query(sql: str) -> list[dict]:
    log.info(f"⚡ [UTILS] Cache MISS → chạy SQL: {sql[:100]}...")
    df = pd.read_sql(text(sql), con=ENGINE)
    return df.to_dict(orient="records")

# Non-parallel execution
def execute_sql(sql: str) -> list[dict]:
    try:
        return _cached_query(sql)
    except Exception as e:
        log.error("⚠️ [UTILS] SQL failed: %s", e)
        raise

# Parallel execution
async def async_execute(sql: str) -> list[dict]:
    """
    Primary RAG utils and SQL executtion.
    1. non-blocking execution through databases/asyncpg or aiomysql
    2. automatic reconnect & fall-back to the sync LRU cache
    """
    try:
        if ENGINE is None:
            log.warning(" [UTILS] DB offline, returning empty rows")
            return []
        if not DATABASE.is_connected:
            await DATABASE.connect()
        results = await DATABASE.fetch_all(query=sql)
        # await DATABASE.disconnect()
        return [dict(row) for row in results]
    except Exception as e:
        log.error("⚠️ [UTILS] async SQL failed: %s", e, " Attempt backup.")
        return execute_sql(sql)   # blocks but never raises here

# ---------- Introspect schema once & cache ----------
@functools.cache
def db_schema() -> dict[str, list[str]]:
    if ENGINE is None:
        log.warning("⚠️ [UTILS] DB unavailable, returning empty schema")
        return {}
    try:
        insp = inspect(ENGINE)
        schema = {tbl: [col["name"] for col in insp.get_columns(tbl)] for tbl in insp.get_table_names()}
        log.info("🎯 [UTILS] DB schema cached: %s", schema.keys())
        return schema
    except Exception as e:
        log.error("⚠️ [UTILS] Schema introspection failed: %s", e)
        return {}
