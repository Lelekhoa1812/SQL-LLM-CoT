# sql.py  ── use Vanna instead of Jina
import os, logging, re
import numpy as np, pandas as pd
import vanna as vn
from google import genai                           

log = logging.getLogger("sql-vanna")
log.info("🚀 Bootstrapping Vanna…")

# ────────────────────────────────────────────────
# 1️. Pick the backend core LLM model
# ────────────────────────────────────────────────
LLM_BACKEND = os.getenv("VN_BACKEND", "gemini").lower() # gemini prefix as default
GEMINI_KEY = os.getenv("GEMINI_FLASH_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"

## **WRAPPER**
from vanna.base import VannaBase
class GeminiVanna(VannaBase):
    def complete_prompt(self, prompt: str, **kwargs) -> str:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(prompt)
            return resp.text.strip()  # No markdown/code fences needed
        except Exception as e:
            log.warning(f"Gemini API call failed: {e}")

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
    log.error("NO GenAI core implemented")
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
