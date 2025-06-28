# sql.py  ── use Vanna instead of Jina
import os, logging, re
import numpy as np, pandas as pd
import vanna as vn
from vanna.base import VannaBase
from google import genai                           
from llm_ut import retry_with_backoff

log = logging.getLogger("sql-vanna")
log.info("🚀 Bootstrapping Vanna…")

# ────────────────────────────────────────────────
# 1️. Pick the backend core LLM model
# ────────────────────────────────────────────────
LLM_BACKEND = os.getenv("VN_BACKEND", "gemini").lower() # gemini prefix as default
GEMINI_KEY = os.getenv("GEMINI_FLASH_API_KEY")
MODEL = "gemini-2.5-flash-preview-04-17"

## **WRAPPER**
class GeminiVanna(VannaBase):
    def __init__(self):
        super().__init__() 
        genai.configure(api_key=os.getenv("GEMINI_FLASH_API_KEY"))
        self.model = genai.GenerativeModel(MODEL)

    @retry_with_backoff(retries=4, delay=1.5)
    def submit_prompt(self, prompt, **kwargs) -> str:
        try:
            if isinstance(prompt, list):
                content = "\n".join(m["content"] for m in prompt)
            else:
                content = str(prompt)
            response = self.model.generate_content(content)
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"[GeminiVanna] Prompt error: {e}")

    # helper used by vanna-core for scoring:
    def score_sql(self, question: str, sql: str) -> float:
        p = (f"Score from 0-1 how well the SQL answers the question.\n"
             f"### Question\n{question}\n### SQL\n{sql}\n"
             "Respond with a single number.")
        try:
            score = float(self.submit_prompt(p).split()[0])
        except Exception:
            score = 0.0
        return max(0.0, min(score, 1.0))
    
    # Minimal viable set of dummy implementations to satisfy ABC
    def add_ddl(self, *args, **kwargs): pass
    def add_documentation(self, *args, **kwargs): pass
    def add_question_sql(self, *args, **kwargs): pass
    def get_related_ddl(self, *args, **kwargs): return []
    def get_related_documentation(self, *args, **kwargs): return []
    def get_similar_question_sql(self, *args, **kwargs): return []
    def get_training_data(self, *args, **kwargs): return []
    def remove_training_data(self, *args, **kwargs): return False
    def system_message(self, message: str): return {"role": "system", "content": message}
    def user_message(self, message: str): return {"role": "user", "content": message}
    def assistant_message(self, message: str): return {"role": "assistant", "content": message}
    def generate_embedding(self, data: str, **kwargs): return [0.0] * 384  # dummy vector

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
from sqlalchemy import create_engine, text
import utils                              

ENGINE = utils.ENGINE
vanna = vn.Vanna(
    llm=llm,
    dialect="mysql",
    get_table_info=lambda: utils.db_schema()
)

# ────────────────────────────────────────────────
# 3️.  A very small “rerank/verify-and-run” helper
# ────────────────────────────────────────────────
def run_and_score(question: str, sqls: list[str]) -> tuple[str, list[dict]]:
    """
    Pick the best SQL using Vanna's built-in scoring,
    then execute via SQLAlchemy.
    """
    if not sqls:
        raise ValueError("No candidate SQL to evaluate")
    # Ask Vanna to rate each proposal (higher is better)
    rated = [
        (vanna.score_sql(question, s), s) for s in sqls
    ]
    rated.sort(reverse=True)
    best_score, best_sql = rated[0]
    log.info("Vanna chose [%0.2f]: %s", best_score, best_sql[:120])
    # Safety-belt: LIMIT 1000 if user forgot
    safe_sql = best_sql if re.search(r"\blimit\b", best_sql, re.I) else best_sql.rstrip(";") + " LIMIT 1000;"
    # Read SQL records
    rows = pd.read_sql(text(safe_sql), ENGINE).to_dict(orient="records")
    return best_sql, rows
