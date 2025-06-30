# Root API: https://binkhoale1812-cpg-chatbot.hf.space/
# DB: https://www.freesqldatabase.com/account/
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bot import QwenBot
from translation import a_vie_to_en, a_en_to_vie
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
    en_trans = a_vie_to_en(req.question) # Translate to english (default lang)
    try:
        sql, rows, answer = await bot.refine_until_valid(
            en_trans, max_loops=10
        )
        log.info(f"[App] ✅ SQL: {sql}\nTop-rows: {rows[:1]}\nAnswer: {answer}")
        vi_trans = a_en_to_vie(answer) # Translate to primary lang
        return {
            "sql": sql,
            "rows": rows,
            "answer": vi_trans
        }
    except Exception as e:
        log.exception("[App] ❌ Failed to answer")
        raise HTTPException(status_code=500, detail=str(e))
