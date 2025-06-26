import json, pathlib, logging, time
log = logging.getLogger("memory-log")
MEM_PATH = pathlib.Path("/app/cache_ltm.json")

def load_ltm() -> dict:
    if MEM_PATH.exists():
        return json.loads(MEM_PATH.read_text())
    return {}

def save_ltm(data: dict):
    MEM_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def add_entry(q: str, sql: str, rows: list[dict]):
    mem = load_ltm()
    mem[q] = {"sql": sql, "rows": rows, "ts": time.time()}
    log.info(f"[Memory] {mem[q]}")
    save_ltm(mem)
