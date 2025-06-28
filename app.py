# Root API: https://binkhoale1812-cpg-chatbot.hf.space/
# DB: https://www.freesqldatabase.com/account/
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bot import QwenBot
from utils import execute_sql

log = logging.getLogger("cpg-chatbot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log.info("🚀 Root app startup...")

# Services
app  = FastAPI()
bot  = QwenBot()

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(req: Query):
    log.info(f"[App] 📥 Incoming question: {req.question}")
    try:
        sql, rows, answer = await bot.refine_until_valid(
            req.question, max_loops=10
        )
        log.info(f"[App] ✅ SQL: {sql}\nTop-rows: {rows[:1]}\nAnswer: {answer}")
        return {
            "sql": sql,
            "rows": rows,
            "answer": answer
        }
    except Exception as e:
        log.exception("[App] ❌ Failed to answer")
        raise HTTPException(status_code=500, detail=str(e))
