# base.py  ── light-weight FAISS-backed memory + prompt helper
import logging, re, hashlib
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import pandas as pd
import faiss                    # pip install faiss-cpu
from sentence_transformers import SentenceTransformer
import memory                    # our Mongo-backed LTM

log   = logging.getLogger("vanna-base")
_EMB  = SentenceTransformer("all-MiniLM-L6-v2")          # 384-d
_DIM  = 384

class VannaBase(ABC):
    """
    What we really need for the vanna-core runtime to work:

    • add_question_sql / add_ddl / add_documentation   – populate a FAISS index
    • similar_qa                                       – RAG retrieval for few-shot SQL
    • get_sql_prompt / extract_sql                     – prompt helpers
    • generate_embedding / (abstract) submit_prompt    – LLM plumbing
    • system|user|assistant_message helpers
    Everything else in the original monster file has been intentionally
    removed – the higher-level logic lives in bot.py and sql.py.
    """

    # --------------------------------------------------------------------- INIT
    def __init__(self, dialect: str = "SQL"):
        self.dialect   = dialect
        self._faiss    = faiss.IndexFlatIP(_DIM)
        self._qa: List[Dict] = []          # [{q,sql,vec}]
        self._ddl:  List[str] = []
        self._docs: List[str] = []

    # -----------------------------------------------------------------  MEMORY
    @staticmethod
    def _embed(text: str) -> np.ndarray:
        return _EMB.encode(text, normalize_embeddings=True)

    def add_question_sql(self, q: str, sql: str):
        vec = self._embed(q).astype("float32")
        self._qa.append({"q": q, "sql": sql, "vec": vec})
        self._faiss.add(vec[np.newaxis, :])
        memory.save({"type": "qa", "q": q, "sql": sql})   # persist to LTM

    def add_ddl(self, ddl: str):           self._ddl.append(ddl);  memory.save({"type":"ddl","ddl":ddl})
    def add_documentation(self, doc: str): self._docs.append(doc); memory.save({"type":"doc","text":doc})

    # -----------------------------------------------------------------  RAG
    def similar_qa(self, question: str, k: int = 4):
        if self._faiss.ntotal == 0:
            return []
        v = self._embed(question).astype("float32")[np.newaxis, :]
        _, idx = self._faiss.search(v, k)
        return [ {"question":self._qa[i]["q"], "sql":self._qa[i]["sql"]} for i in idx[0] ]

    # ------------------------------------------------------------  PROMPT UTILS
    @staticmethod
    def extract_sql(text: str) -> str:
        """Return the last SQL-looking chunk from *text*."""
        pats = [r"```sql\s*([\s\S]+?)```",
                r"\bSELECT\b[\s\S]+?;",
                r"\bWITH\b[\s\S]+?;",
                r"\bCREATE\s+TABLE\b[\s\S]+?;"]
        for p in pats:
            hit = re.findall(p, text, flags=re.I)
            if hit:
                return hit[-1].strip()
        return text.strip()

    def get_sql_prompt(self, question: str) -> str:
        ctx   = "\n".join(self._ddl[:4])
        docs  = "\n".join(self._docs[:2])
        shots = "\n\n".join(f"-- {ex['question']}\n{ex['sql']}"
                            for ex in self.similar_qa(question, k=3))
        return (
            f"You are a {self.dialect} expert.\n"
            f"### TABLES ###\n{ctx}\n"
            f"### DOCS ###\n{docs}\n"
            f"### EXAMPLES ###\n{shots}\n\n"
            f"### QUESTION ###\n{question}\n\n"
            "Respond with *only* the executable SQL."
        )

    # -------------------------------------------------------  LLM INTERFACING
    def system_message(self, x: str):     return {"role":"system",    "content":x}
    def user_message(self,   x: str):     return {"role":"user",      "content":x}
    def assistant_message(self,x:str):    return {"role":"assistant", "content":x}

    def generate_embedding(self, txt: str): return self._embed(txt).tolist()

    @abstractmethod
    def submit_prompt(self, prompt: str) -> str:
        """Concrete subclasses (GeminiVanna, OpenAIVanna, …) override this."""
