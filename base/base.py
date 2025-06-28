# base.py  ── light-weight FAISS-backed RAG helper
import re, logging, numpy as np, faiss
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import memory                         # our Mongo LTM wrapper

log   = logging.getLogger("vanna-base")
_EMB  = SentenceTransformer("all-MiniLM-L6-v2")   # 384-d
_DIM  = 384

class VannaBase(ABC):
    def __init__(self, dialect: str = "SQL"):
        self.dialect     = dialect
        self._index      = faiss.IndexFlatIP(_DIM)
        self._examples   = []      # [{q,sql,vec}]
        self._ddl, self._docs = [], []

    # ---------- util ---------------------------------------------------------
    def _embed(self, txt: str) -> np.ndarray:
        return _EMB.encode(txt, normalize_embeddings=True)

    # ---------- persisting / RAG --------------------------------------------
    def add_question_sql(self, q: str, sql: str):
        vec = self._embed(q).astype("float32")
        self._examples.append({"q": q, "sql": sql, "vec": vec})
        self._index.add(vec[None, :])
        memory.save({"type": "qa", "q": q, "sql": sql})     # LTM

    def add_ddl(self, ddl: str):           self._ddl.append(ddl);  memory.save({"type":"ddl","ddl":ddl})
    def add_documentation(self, doc: str): self._docs.append(doc); memory.save({"type":"doc","text":doc})

    def similar_qa(self, question: str, k=3):
        if self._index.ntotal == 0: return []
        D, I = self._index.search(self._embed(question)[None, :], k)
        return [ {"question":self._examples[i]["q"], "sql":self._examples[i]["sql"]}
                 for i in I[0] if i != -1 ]

    # ---------- prompt helpers ----------------------------------------------
    @staticmethod
    def extract_sql(text: str) -> str:
        pats = [r"```sql[\s\S]+?```", r"\bWITH\b[\s\S]+?;", r"\bSELECT\b[\s\S]+?;",
                r"\bCREATE\s+TABLE\b[\s\S]+?;"]
        for p in pats:
            m = re.findall(p, text, flags=re.I)
            if m: return re.sub(r"```sql|```", "", m[-1]).strip()
        return text.strip()

    def get_sql_prompt(self, question: str) -> str:
        ctx   = "\n".join(self._ddl[:4])
        docs  = "\n".join(self._docs[:2])
        shots = "\n\n".join(f"-- {ex['question']}\n{ex['sql']}"
                            for ex in self.similar_qa(question))
        return (
            f"You are a {self.dialect} expert.\n"
            f"### TABLES ###\n{ctx}\n"
            f"### DOCS ###\n{docs}\n"
            f"### EXAMPLES ###\n{shots}\n\n"
            f"### QUESTION ###\n{question}\n\n"
            "Respond with ONLY the executable SQL."
        )

    # ---------- LLM plumbing -------------------------------------------------
    @abstractmethod
    def submit_prompt(self, prompt: str) -> str: ...

    def system_message(self, x):  return {"role":"system","content":x}
    def user_message(self, x):    return {"role":"user","content":x}
    def assistant_message(self,x):return {"role":"assistant","content":x}
