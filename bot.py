# bot.py  ── Qwen (reasoning)  ↔  Gemini-Vanna (SQL RAG)
import os, re, asyncio, logging
from gradio_client import Client
from utils   import execute_sql
from sql     import run_and_score, vanna          # vanna = GeminiVanna instance
import translation as tr
import memory
from llm_ut  import retry_with_backoff

log = logging.getLogger("qwen-bot")
log.info("🚀 Booting Qwen assistant")

SYSTEM_PROMPT = (
    "You are a retail-data analysis assistant. "
    "Think step-by-step, are proficient in SQL, "
    "NEVER reveal your thoughts — only the final answer."
)

HF_SPACE = "mikeee/qwen-7b-chat"
API_NAME = "/user"

class QwenBot:
    def __init__(self):
        token    = os.getenv("HF_TOKEN")
        self.client = Client(HF_SPACE, hf_token=token or None)
        self.chat_history : list[tuple[str, str]] = []
        asyncio.run(self._cold_start())

    # ──────────── Low-level LLM wrappers ────────────
    @retry_with_backoff(retries=4, delay=1.5)
    def _llm(self, prompt: str) -> str:
        reply, hist = self.client.predict(prompt, self.chat_history, api_name=API_NAME)
        self.chat_history = hist
        return reply

    @retry_with_backoff(retries=4, delay=1.5)
    def _llm_no_mem(self, prompt: str) -> str:
        reply, _ = self.client.predict(prompt, [], api_name=API_NAME)
        return reply

    async def _cold_start(self, rounds: int = 10):
        """
        Bootstraps QwenBot before user input:
        - Load LTM into STM
        - Generate synthetic QA pairs using CoT + Vanna
        - Run SQL, reason, and save back into memory
        """
        log.info(f"[Cold Start] started!")
        # ───── Step 1: Load prior memory into STM & chat_history ─────
        prior_memories = memory.retrieve_ltm("", top_k=50)
        for doc in prior_memories:
            q, sql, ans = doc["question"], doc["sql"], doc.get("answer", "")
            memory.add_stm(q, {"sql": sql, "rows": doc["rows"], "answer": ans})
            self.chat_history.append((f"(memory) {q}", ans))
        log.info("✅ Loaded %d memory entries into STM and chat_history", len(prior_memories))
        # ───── Step 2: Describe current schema using Qwen ─────
        schema = memory.db_schema() if hasattr(memory, 'db_schema') else {}  # fallback
        schema_txt = "\n".join(f"{t}({', '.join(cols)})" for t, cols in schema.items())
        intro = f"Database schema:\n{schema_txt}\n\nSummarise each table and typical queries."
        summary = await asyncio.to_thread(self._llm, intro)
        memory.add_ltm_entry("__SCHEMA_SUMMARY__", "", [], summary)
        self.chat_history.append(("(schema summary)", summary))
        log.info("🧠 Added schema summary to LTM + STM")
        # ───── Step 3: Chain-of-thought enrichment ─────
        for i in range(1, rounds + 1):
            log.info(f"[ColdStart-Round {i}] Generating CoT questions…")
            cot_prompt = (
                "Generate several realistic business questions about sales data. "
                "Cover time trends, geographic differences, product categories, etc. "
                "For each question, include a SQL query below it."
            )
            cot_raw = await asyncio.to_thread(self._llm, cot_prompt)
            questions = re.findall(r"Question:\s*(.+)", cot_raw, re.I)
            sqls      = re.findall(r"(SELECT .*?;)", cot_raw, re.I | re.S)
            # Save valid LTMs
            memory.add_ltm_entry(f"__COT_{i}__", "", [], cot_raw)
            log.info(f"CoT-{i}: Generated {len(questions)} Q–SQL candidates")
            for q, raw_sql in zip(questions, sqls):
                try:
                    # Rerank (only 1 candidate here)
                    best_sql, rows = run_and_score(q, [raw_sql])
                    if not rows: continue
                    # Use Qwen to reason over output
                    reason_prompt = (
                        f"Q: {q}\nSample result:\n{str(rows[:5])}\n"
                        "What can you deduce from this data?"
                    )
                    rationale = await asyncio.to_thread(self._llm, reason_prompt)
                    # Save all to memory
                    answer_pack = {"sql": best_sql, "rows": rows, "answer": rationale}
                    memory.add_stm(q, answer_pack)
                    memory.add_ltm_entry(q, best_sql, rows, rationale)
                    self.chat_history.append((q, rationale))
                    log.info(f"✅ [COT-{i}] Stored: {q[:40]}...")
                except Exception as e:
                    log.warning(f"❌ [COT-{i}] Failed for Q: {q[:30]} — {e}")
        # Logs
        log.info(f"[Cold Start] completed: {rounds} CoT rounds finished")

    # ──────────── Helper: ask Gemini-Vanna for 1 SQL ────────────
    def _vanna_sql(self, question_en: str) -> str:
        prompt = vanna.get_sql_prompt(question_en)
        raw    = vanna.submit_prompt(prompt)
        return vanna.extract_sql(raw)

    # ──────────── Helper: ask Qwen to brainstorm N SQLs ────────────
    async def _brainstorm_sqls(self, question_en: str, n: int = 6):
        prompt = (
            f"Give {n} distinct SQL queries that could answer the question:\n"
            f"\"{question_en}\"\nOnly return the SQL, no explanation."
        )
        raw = await asyncio.to_thread(self._llm, prompt)
        return re.findall(r"SELECT .*?;", raw, flags=re.I | re.S)

    # ──────────── The important bit: refine_until_valid ────────────
    async def refine_until_valid(self, question_vi: str, max_loops: int = 5):
        """Try STM ➜ LTM ➜ Vanna/Qwen loops until an executable SQL + answer."""
        question_en = await tr.a_vie_to_en(question_vi)
        # 0) Cached?
        if (hit := memory.get_stm(question_en)):
            log.info("[STM] hit")
            return hit["sql"], hit["rows"], hit["answer"]
        if (ltm := memory.retrieve_ltm(question_en, 1)):
            doc = ltm[0]; memory.add_stm(question_en, doc)
            log.info("[LTM] hit")
            return doc["sql"], doc["rows"], doc["answer"]
        # Stack candidates
        last_error = ""
        for attempt in range(1, max_loops + 1):
            cand_sqls = []
            # (a) Ask Vanna
            try:
                cand_sqls.append(self._vanna_sql(question_en))
            except Exception as e:
                log.warning("Vanna failed: %s", e)
            # (b) Brain-storm with Qwen (and optionally feed previous error)
            if last_error:
                self.chat_history.append((f"The last SQL failed: {last_error}", ""))
            cand_sqls += await self._brainstorm_sqls(question_en)
            # Remove dupes while preserving order
            seen, uniq = set(), []
            for sql in cand_sqls:
                key = re.sub(r"\s+", " ", sql.strip().lower())
                if key not in seen:
                    seen.add(key); uniq.append(sql)
            cand_sqls = uniq[:10]           # keep it short for scorer
            # (c) Pick best by Vanna scorer + verify
            try:
                best_sql, rows = run_and_score(question_en, cand_sqls)
                if not rows: raise ValueError("0 rows returned")
                answer = await self._craft_final_answer(question_en, rows)
                payload = {"sql": best_sql, "rows": rows, "answer": answer}
                memory.add_stm(question_en, payload)
                memory.add_ltm_entry(question_en, best_sql, rows, answer)
                log.info("Solved on attempt %d", attempt)
                return best_sql, rows, answer
            except Exception as e:
                last_error = str(e)
                log.warning("Attempt %d failed: %s", attempt, last_error)
        # Error
        raise RuntimeError("❌ Could not obtain a valid SQL after many tries.")

    # ──────────── Compose short natural-language answer ────────────
    async def _craft_final_answer(self, question_en: str, rows):
        sample = str(rows[:8])
        prompt = (
            f"Q: {question_en}\n"
            f"Rows sample: {sample}\n\n"
            "Give a concise factual answer (one sentence)."
        )
        return await asyncio.to_thread(self._llm_no_mem, prompt)
