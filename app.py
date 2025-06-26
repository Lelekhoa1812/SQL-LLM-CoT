# Root API: https://binkhoale1812-cpg-chatbot.hf.space/
# DB: https://www.freesqldatabase.com/account/
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bot import QwenBot
from sql import SQLReranker
from utils import execute_sql

log = logging.getLogger("cpg-chatbot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log.info("🚀 Root app startup...")

# Services
app  = FastAPI()
bot  = QwenBot()
rank = SQLReranker()

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(req: Query):
    try:
        sql, rows, answer = await bot.refine_until_valid(
            req.question, execute_sql, rank.rerank, max_loops=5
        )
        log.info(f"[App] sql: {sql}, rows: {rows[:5]}, answer: {answer}")
        return {"sql": sql, "rows": rows, "answer": answer}
    except Exception as e:
        log.error("[App] ❌ %s", e)
        raise HTTPException(status_code=500, detail=str(e))
