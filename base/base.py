# base/base.py  ── light-weight FAISS-backed RAG helper
import os, re, logging, json, numpy as np, faiss
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
import memory                         # our Mongo LTM wrapper

log   = logging.getLogger("vanna-base")
_EMB  = SentenceTransformer("all-MiniLM-L6-v2")   # 384-d
_DIM  = 384

class VannaBase(ABC):
    """
    Minimal Vanna-style abstraction:
    • In-memory FAISS vector index
    • Few-shot store (QA), DDL, docs
    • Helper to build SQL-only prompts
    """

    def __init__(self, dialect: str = "SQL"):
        self.dialect      = dialect
        self._index       = faiss.IndexFlatIP(_DIM)
        self._examples: List[Dict] = []      # [{q, sql, vec}]
        self._ddl:  List[str] = []
        self._docs: List[str] = []

    # ---------- Embedding ---------------------------------------------------------
    def _embed(self, txt: str) -> np.ndarray: # shape (384,...)
        return _EMB.encode(txt, normalize_embeddings=True)

    # ---------- RAG memory---------------------------------------------------------
    def add_question_sql(self, q: str, sql: str):
        norm_q = re.sub(r"[\W_]+", "", q.lower())
        if any(re.sub(r"[\W_]+", "", ex["q"].lower()) == norm_q for ex in self._examples):
            log.info(f"[FAISS] Skipping duplicate: {q[:60]}")
            return
        vec = self._embed(q).astype("float32")
        self._examples.append({"q": q, "sql": sql, "vec": vec})
        self._index.add(vec[None, :])

    def add_ddl(self, ddl: str):           self._ddl.append(ddl);  
    def add_documentation(self, doc: str): self._docs.append(doc); 

    def similar_qa(self, question: str, k: int = 3) -> List[Dict]:
        if self._index.ntotal == 0:
            return []
        D, I = self._index.search(self._embed(question)[None, :], k)
        return [ {
                "question": self._examples[i]["q"],
                "sql": self._examples[i]["sql"],
                }
            for i in I[0] if i != -1
        ]
    
    def save_index(self, path: str = "faiss_index.bin"):
        faiss.write_index(self._index, path)

    def load_index(self, path: str = "faiss_index.bin"):
        if os.path.exists(path):
            self._index = faiss.read_index(path)


    # ---------- Prompt helpers ----------------------------------------------
    @staticmethod
    def extract_sql(text: str) -> str:
        """
        Pull the last plausible SQL snippet from `text`.
        """
        pats = [
            r"```sql\s*([\s\S]+?)```",
            r"\bWITH\b[\s\S]+?;",
            r"\bSELECT\b[\s\S]+?;",
            r"\bCREATE\s+TABLE\b[\s\S]+?;",
        ]
        for p in pats:
            m = re.findall(p, text, flags=re.I)
            if m:
                return re.sub(r"```sql|```", "", m[-1]).strip()
        return text.strip()

    # ---- NEW: accepts the four-arg Vanna signature (all optional) -------
    def get_sql_prompt(
        self,
        question: str,
        *,
        initial_prompt: Optional[str] = None,
        question_sql_list: Optional[List[Dict]] = None,
        ddl_list: Optional[List[str]] = None,
        doc_list: Optional[List[str]] = None,
        shots: Optional[List[Dict]] = None,
    ) -> str:
        """
        Build a single text prompt containing:
        • opening system role (initial_prompt or default)
        • DDL context
        • documentation snippets
        • few-shot examples
        • the QUESTION block
        """
        base_header = initial_prompt or f"You are a {self.dialect} expert."
        ddl_text  = "\n".join(ddl_list or self._ddl[:4])
        doc_text  = "\n".join(doc_list or self._docs[:2])
        # Choose examples priority: explicit list → 'shots' alias → self.similar_qa()
        examples = question_sql_list or shots or self.similar_qa(question)
        examples_txt = "\n\n".join(
            f"-- {ex['question']}\n{ex['sql']}"
            + (f"\n-- Sample Output:\n{json.dumps(ex.get('rows', [])[:2], indent=2)}"
            if ex.get("rows") else "")
            for ex in examples if ex.get("sql")
        )
        # Return executable SQL formatted JSON
        return (
            f"{base_header}\n"
            f"### TABLES ###\n{ddl_text}\n"
            f"### DOCS ###\n{doc_text}\n"
            f"### EXAMPLES ###\n{examples_txt}\n\n"
            f"### QUESTION ###\n{question}\n\n"
            "Respond with ONLY the executable SQL."
        )
    
    # ---------- LLM plumbing -------------------------------------------------
    @abstractmethod
    def submit_prompt(self, prompt: str) -> str: ...

    def system_message(self, x):  return {"role":"system","content":x}
    def user_message(self, x):    return {"role":"user","content":x}
    def assistant_message(self,x):return {"role":"assistant","content":x}
