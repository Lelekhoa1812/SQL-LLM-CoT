# bot.py
from gradio_client import Client
import asyncio, logging, re, os
from utils import db_schema, execute_sql
from sql   import SQLReranker
import memory
import translation as tr
from llm_ut import retry_with_backoff

log = logging.getLogger("qwen-bot")
log.info("🚀 Starting Qwen bot...")

# ---------- constants -------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a retail-data analysis assistant. "
    "You think step-by-step (chain-of-thought) and are proficient in SQL. "
    "NEVER reveal your thoughts – only give the final answer."
)

HF_SPACE = "mikeee/qwen-7b-chat"        # English-capable Qwen client
API_NAME  = "/user"

class QwenBot:
    def __init__(self):
        token = os.getenv("HF_TOKEN")
        self.client = Client(HF_SPACE, hf_token=token or None)
        self.chat_history: list[tuple[str, str]] = []  # STM-style memory
        self.reranker = SQLReranker()
        self.system_prompt = SYSTEM_PROMPT
        asyncio.run(self._build_up())
    
    # ---------- low-level wrappers ----------------------------------------
    @retry_with_backoff(retries=4, delay=1.5) # Retry if error persist
    def _llm_no_mem(self, prompt_en: str) -> str:
        """One-off call without history (used for final summaries)."""
        reply, _ = self.client.predict(message=prompt_en, chat_history=[], api_name=API_NAME)
        return reply
    
    @retry_with_backoff(retries=4, delay=1.5) # Retry if error persist
    def _llm(self, prompt_en: str) -> str:
        """LLM call that keeps self.chat_history (STM)."""
        reply, hist = self.client.predict(
            message=prompt_en,
            chat_history=self.chat_history,
            api_name=API_NAME
        )
        self.chat_history = hist
        return reply

    # ---------- BUILD-UP PHASE -------------------------------------------
    async def _build_up(self, rounds: int = 10):
        """
        1) Load existing LTM entries and seed STM (chat_history)
        2) Describe schema
        3) Repeat CoT rounds: propose Q→SQL, rerank, exec, reason, save LTM
        """
        # 1) seed from LTM
        existing = memory.retrieve_ltm("", top_k=100)
        for e in existing:
            memo = f"Memory: '{e['question']}' -> SQL: {e['sql']}"
            self.chat_history.append((memo, e['answer'] or ""))
            log.info(f"[Qwen - Seed] Memory: '{e['question']}' → SQL: {e['sql']}")
        
        # 2) describe schema
        schema = db_schema()
        schema_txt = "\n".join(f"{t}({', '.join(cols)})" for t, cols in schema.items())
        # Prompt engineering
        intro = (
            f"Database schema:\n{schema_txt}\n\n"
            "Summarise each table, relationships and common query types."
        )
        summary = await asyncio.to_thread(self._llm, intro)
        memory.add_ltm_entry("__SCHEMA_SUMMARY__", "", [], summary)
        log.info(f"[Qwen - Build-up 0] Saved schema summary {summary}")
        
        # 3) CoT enrichment rounds
        for i in range(1, rounds + 1):
            log.info(f"[Build-up {i}] Thinking of new Q→SQL pairs…")
            cot_prompt = (
                "Generate several advanced business questions about sales data "
                "(time trends, regions, categories…). For each, propose a SQL query."
            )
            cot = await asyncio.to_thread(self._llm, cot_prompt)
            # extract questions & SQLs
            qs  = re.findall(r"Question:\s*(.+)", cot, flags=re.I)
            sqls = re.findall(r"(SELECT .*?;)", cot, flags=re.I|re.S)
            memory.add_ltm_entry(f"__COT_{i}__", "", [], cot)
            log.info(f"[Qwen - Build-up {i}] CoT: {cot}")
            # for each Q→SQL, rerank, exec & reason
            for q, raw_sql in zip(qs, sqls):
                best_sql = self.reranker.rerank(q, [raw_sql])  # single candidate
                rows     = execute_sql(best_sql)
                # feed back to model to reason on results
                reason_prompt = (
                    f"Here are the first rows of the query result:\n{rows[:5]}\n\n"
                    "Provide any additional insight you can derive."
                )
                reasoning = await asyncio.to_thread(self._llm, reason_prompt)
                # persist
                memory.add_ltm_entry(q, best_sql, rows, reasoning)
                log.info(f"[Qwen - Build-up {i}] Saved Q→SQL→Reason for: {q[:30]}…")
        log.info(f"[Qwen - Build-up] ✅ Finish {rounds} of enrichment")

    # ---------- RUNTIME PHASE --------------------------------------------
    # Phase 1: generate SQL thoughts
    async def generate_sql_thoughts(self, question: str):
        """
        Generate up to 6 SQL candidates for a new user question.
        """
        prompt = (
            "Based on the schema and your memory, propose up to 6 SQL queries "
            f"that could answer: {question}"
        )
        raw = await asyncio.to_thread(self._llm, prompt)
        sqls = re.findall(r"(SELECT .*?;)", raw, flags=re.IGNORECASE|re.DOTALL)
        log.info(f"[Qwen - SQL] {sqls}")
        return raw, [s.replace("\n"," ").strip() for s in sqls]

    # Phase 2: generate concise answer
    async def generate_answer(self, question: str, data: list[dict], raw: str) -> str:
        """
        Craft a concise VN answer from raw SQL results.
        """
        preview = str(data[:10])
        prompt = (
            f"Question: {question}\nSample result: {preview}\n\n"
            "Provide a concise, factual answer."
        )
        res = await asyncio.to_thread(self._llm_no_mem, prompt)
        log.info(f"[Qwen - Answer] {res}")
        return res

    # Phase 3: refine + caching
    async def refine_until_valid(
        self, question_vi: str, exec_fn, rerank_fn, max_loops: int = 3
    ):
        # → English
        question_en = await tr.a_vie_to_en(question_vi)
        """
        1) STM hit?
        2) LTM hit?
        3) Fallback: loop Q→SQL→rerank→exec→answer up to max_loops
        """
        # STM
        if stm := memory.get_stm(question_en):
            log.info("[STM] re-using cached")
            return stm["sql"], stm["rows"], stm["answer"]
        # LTM
        ltms = memory.retrieve_ltm(question_en, top_k=1)
        if ltms:
            e = ltms[0]
            memory.add_stm(question_en, e)
            log.info("[LTM] re-using memory SQL")
            return e["sql"], e["rows"], e["answer"]
        # fallback
        for attempt in range(1, max_loops+1):
            raw, cands = await self.generate_sql_thoughts(question_en)
            best = rerank_fn(question_en, cands)
            try:
                rows = exec_fn(best)
                if rows:
                    ans = await self.generate_answer(question_en, rows, raw)
                    resp = {"sql": best, "rows": rows, "answer": ans}
                    memory.add_stm(question_en, resp)
                    memory.add_ltm_entry(question_en, best, rows, ans)
                    return best, rows, ans
            except Exception as e:
                log.warning(f"[Fallback {attempt}] SQL error: {e}")
        raise RuntimeError("Không thể tạo SQL hợp lệ sau nhiều vòng.")