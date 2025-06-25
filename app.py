import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bot import QwenBot
from sql import SQLReranker
from utils import execute_sql

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cpg-chatbot")
app = FastAPI()
logger.info("🚀 Starting Chatbot API server...")

# Initialize models
bot = QwenBot()
reranker = SQLReranker()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query(request: QueryRequest):
    question = request.question
    # 1) Generate SQL candidates with chain-of-thought
    thinking, sql_candidates = await bot.generate_sql_thoughts(question)
    logger.info(f"Generated SQL candidates: {sql_candidates}")

    # 2) Rerank SQL candidates
    best_sql = reranker.rerank(question, sql_candidates)
    logger.info(f"Selected best SQL: {best_sql}")

    # 3) Execute SQL against the DB
    try:
        result = execute_sql(best_sql)
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 4) Generate final answer using LLM
    answer = await bot.generate_answer(question, result, thinking)
    return {
        "sql": best_sql,
        "result": result,
        "answer": answer
    }