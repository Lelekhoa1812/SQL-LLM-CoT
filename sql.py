# sql.py
import os, logging, re
from typing import List
import numpy as np, pandas as pd
# LLM
import vanna as vn
from base.base import VannaBase  # Custom lightweight Vanna
from google import genai  
# Util services
from llm_ut import retry_with_backoff
from memory import add_sql_pair, retrieve_sql, get_stm, add_stm
import utils
from sqlalchemy import text as sa_text
from rotator import RotatingGeminiClient

log = logging.getLogger("sql-vanna")
log.info("🚀 Bootstrapping Vanna…")

# ────────────────────────────────────────────────
# 1️. Pick the backend core LLM model
# ────────────────────────────────────────────────
LLM_BACKEND = os.getenv("VN_BACKEND", "gemini").lower() # gemini prefix as default
MODEL = "gemini-2.5-flash-preview-04-17"

## **WRAPPER**
class GeminiVanna(VannaBase):
    def __init__(self):
        super().__init__() 
        self.client = RotatingGeminiClient
        self.dialect = "mysql"
        schema = utils.db_schema() # preload DDL from schema so get_sql_prompt() has context
        for t, cols in schema.items():
            self.add_ddl(f"{t}({', '.join(cols)})")
        self.model = MODEL

    @retry_with_backoff(retries=4, delay=1.5)
    def submit_prompt(self, prompt, **kwargs) -> str:
        """`prompt` is plain text built by get_sql_prompt()."""
        content = "\n".join(p["content"] for p in prompt) if isinstance(prompt, list) else str(prompt)
        resp = self.client.models.generate_content(model=self.model, contents=[{"role": "user", "parts": [{"text": content}]}])
        return resp.text.strip()

    # helper used by vanna-core for scoring:
    def score_sql(self, question: str, sql: str) -> float:
        p = (f"Score from 0-1 how well the SQL answers the question.\n"
             f"### Question\n{question}\n### SQL\n{sql}\n"
             "Respond with a single number.")
        try:
            match = re.search(r"([01](?:\.\d+)?)", self.submit_prompt(p))
            score = float(match.group(1)) if match else 0.0
        except Exception:
            score = 0.0
        return max(0.0, min(score, 1.0))
    
    def extract_sql(self, text: str) -> str:
        return super().extract_sql(text)

    def similar_qa(self, q: str, k: int = 3):
        """
        Retrieve top-k similar (question, SQL) pairs from LTM via memory vector search.
        """
        docs = retrieve_sql(q, top_k=k) or super().similar_qa(q, k)
        return [
            {"question": d.get("question", ""), "sql": d.get("sql", "")}
            for d in docs if "question" in d and "sql" in d
        ]

    def add_question_sql(self, q: str, sql: str):
        """
        Execute SQL safely; store to memory even if it returns 0 rows
        (rows retained for future examples).
        """
        rows = []
        try:
            rows = pd.read_sql(sa_text(sql), utils.ENGINE).to_dict(orient="records")
        except Exception as e:
            log.warning(f"[add_question_sql] SQL execution failed: {e}")
            rows = []
        super().add_question_sql(q, sql)
        add_sql_pair(q, sql, rows, "")

# **Strongly preferable**
if LLM_BACKEND == "openai":
    from vanna.openai import OpenAIVanna as VannaLLM
    llm = VannaLLM(api_key=os.getenv("OPENAI_API_KEY"))
elif LLM_BACKEND == "cohere":
    from vanna.cohere import CohereVanna as VannaLLM
    llm = VannaLLM(api_key=os.getenv("COHERE_API_KEY"))

# **Usable but less effective wrapper**
elif LLM_BACKEND == "gemini":
    llm = GeminiVanna()
else:
    log.error("[SQL] NO GenAI core implemented")
    class CustomVanna(VannaBase):
        def complete_prompt(self, prompt, **kwargs):
            """
            Call your favourite endpoint here (Groq, Gemini, LM-Studio …)
            Must return the raw assistant string.
            """
            raise NotImplementedError("Plug your LLM here")
    llm = CustomVanna()

# ────────────────────────────────────────────────
# 2️.  Create the Vanna agent bound to our DB
# ────────────────────────────────────────────────
ENGINE = utils.ENGINE
vanna = llm

# ────────────────────────────────────────────────
# 3️.  A very small “rerank/verify-and-run” helper
# ────────────────────────────────────────────────
async def run_and_score(question: str, sqls: List[str]):
    """
    • score each SQL with Vanna
    • pick best
    • run via SQLAlchemy
    • return (best_sql, rows)
    """
    if not sqls: raise ValueError("No candidate SQL provided")
    rated = [(vanna.score_sql(question, s), s) for s in sqls]
    rated.sort(reverse=True)
    best_score, best_sql = rated[0]
    log.info("Vanna picked [%0.2f]: %s", best_score, best_sql[:120])
    # add LIMIT 1000 safeguard if missing
    safe_sql = (
        best_sql if re.search(r"\blimit\b", best_sql, re.I)
        else best_sql.rstrip(";") + " LIMIT 1000;"
    )
    # Read SQL save records to dict
    rows = await utils.async_execute(safe_sql)
    return best_sql, rows

