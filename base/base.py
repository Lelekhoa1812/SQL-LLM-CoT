# base.py  ── *minimal* FAISS-backed memory + prompt helper
import logging, os, re, hashlib
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import pandas as pd
import faiss                           # pip install faiss-cpu
from sentence_transformers import SentenceTransformer

log   = logging.getLogger("vanna-base")
EMBED = SentenceTransformer("all-MiniLM-L6-v2")          # 384-d

# --------------------------------------------------------------------------------------
class VannaBase(ABC):
    """
    What we actually need:
      • embed()        –> FAISS vector store
      • get_sql_prompt / extract_sql
      • submit_prompt  – abstract, implemented by subclass (Gemini / OpenAI / …)
    Everything else in the huge original file was deleted.
    """
    DIM = 384

    def __init__(self, dialect: str = "SQL"):
        self.dialect   = dialect
        self.idx       = faiss.IndexFlatIP(self.DIM)
        self.qa_store: List[Dict] = []   # [{q,sql,vec}]
        self.ddl:  List[str]   = []
        self.docs: List[str]   = []
        self.static_doc = ""

    # ---------- tiny persistence helpers ------------------------------------------------
    def add_question_sql(self, q: str, sql: str) -> None:
        v = self._embed(q)
        self.qa_store.append({"q": q, "sql": sql, "vec": v})
        self.idx.add(np.expand_dims(v,0))

    def add_ddl(self, ddl: str):              self.ddl.append(ddl)
    def add_documentation(self, doc: str):    self.docs.append(doc)

    def similar_qa(self, question: str, k: int = 5):
        if self.idx.ntotal == 0:
            return []
        v = self._embed(question).reshape(1,-1)
        _, I = self.idx.search(v, k)
        return [ {"question":self.qa_store[i]["q"], "sql":self.qa_store[i]["sql"]}
                 for i in I[0] ]

    # ---------- public helpers used by sql.py / bot.py ----------------------------------
    @staticmethod
    def extract_sql(text: str) -> str:
        """
        Grab the last SQL-looking chunk from an LLM answer.
        """
        patterns = [
            r"```sql\s*([\s\S]*?)```",
            r"\bSELECT\b[\s\S]*?;",
            r"\bWITH\b[\s\S]*?;",
            r"\bCREATE\s+TABLE\b[\s\S]*?;"
        ]
        for p in patterns:
            m = re.findall(p, text, flags=re.I)
            if m:
                return m[-1].strip()
        return text.strip()

    def get_sql_prompt(self, question: str) -> str:
        ctx  = "\n".join(self.ddl[:5])  # you can choose a smarter selection
        docs = "\n".join(self.docs[:3])
        shots= "\n\n".join(f"-- {ex['question']}\n{ex['sql']}"
                           for ex in self.similar_qa(question, k=3))
        prompt = (
            f"You are a {self.dialect} expert.\n"
            f"### TABLES ###\n{ctx}\n"
            f"### DOCS ###\n{docs}\n"
            f"### EXAMPLES ###\n{shots}\n\n"
            f"### QUESTION ###\n{question}\n\n"
            "Write only the SQL you would run."
        )
        return prompt

    # ---------- abstract API ------------------------------------------------------------
    @abstractmethod
    def submit_prompt(self, prompt: str) -> str: ...

    # ---------- internals ----------------------------------------------------------------
    @staticmethod
    def _embed(text: str) -> np.ndarray:
        return EMBED.encode(text, normalize_embeddings=True)
