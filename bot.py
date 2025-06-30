# bot.py  ── Qwen (reasoning)  ↔  Gemini-Vanna (SQL RAG)
import os, re, asyncio, logging, json
from utils   import execute_sql, db_schema
from sql     import run_and_score, vanna # vanna = GeminiVanna instance
import translation as tr
import memory
from google import genai   
from google.genai.types import Content, Part
from llm_ut  import retry_with_backoff
from rotator import RotatingGeminiClient

log = logging.getLogger("qwen-bot")
log.info("🚀 Booting Qwen assistant")

SYSTEM_PROMPT = (
    "You are a retail-data analysis assistant. "
    "Think step-by-step, proficient in SQL, "
    "NEVER reveal your thoughts — only the final answer, "
    "Convert this prompt to SQL backed up by historical data (if applicable)."
)

# ──────────── Gemini Config ────────────
genai_client = RotatingGeminiClient() # Switch between Gemini clients when one not available
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"

def _clean_md(text: str) -> str:
    """Strip markdown fences"""
    if "```" in text:
        match = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match[0].strip()
    return text.strip()

class QwenBot:
    def __init__(self):
        self.chat_history: list[Content] = []
        asyncio.run(self._cold_start())

    @retry_with_backoff(retries=3, delay=1.5)
    def _llm(self, prompt: str) -> str:
        """Use Gemini with memory"""
        # Clip history down to minimize max token input to Gemini
        MAX_HISTORY = 30 # 30 entries max
        # User prompt + history 
        contents = [
            Content(role="model", parts=[Part(text=SYSTEM_PROMPT)]),
            *self.chat_history[-MAX_HISTORY:],  # preserve memory
            Content(role="user", parts=[Part(text=prompt)])
        ]
        try:
            rsp = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents
            )
            self.chat_history.append(Content(role="model", parts=[Part(text=f"(question): {prompt} - (answer) {rsp.text}")]))
            return _clean_md(rsp.text)
        except Exception as e:
            log.error(f"[Gemini] memory LLM failed: {e}")
            raise

    @retry_with_backoff(retries=4, delay=1.5)
    def _llm_no_mem(self, prompt: str) -> str:
        """Stateless one-shot call"""
        try:
            rsp = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[Content(role="user", parts=[Part(text=prompt)])]
            )
            return _clean_md(rsp.text)
        except Exception as e:
            log.error(f"[Gemini] no-mem LLM failed: {e}")
            raise

    def reset_history(self):
        self.chat_history = []
        logging.info("🧼 Cleared chat_history")

    def _parse_cot_fallback(text: str) -> list[dict]:
        questions = re.findall(r"(?i)question:\s*(.+?)(?:\n|$)", text)
        sqls = re.findall(r"```sql\s*(.*?)```|(?:SELECT[\s\S]+?;)", text, flags=re.I)
        clean_sqls = [s if isinstance(s, str) else s[0] for s in sqls]
        return [{"question": q.strip(), "sql": s.strip()} for q, s in zip(questions, clean_sqls)]

    @retry_with_backoff(retries=5, delay=1)
    async def _cold_start(self, rounds: int = 10):
        """
        Bootstraps QwenBot before user input:
        - Loads LTM into STM
        - Summarizes schema
        - Generates question-SQL pairs using Gemini
        - Executes, reranks, reasons over them
        - Saves only successful ones into LTM and STM
        """
        self.reset_history()
        memory.clear_all()
        log.info("[Cold Start] started")
        # ───── Step 1: Load existing memory into STM/chat_history ─────
        prior_memories = memory.retrieve_ltm("", top_k=50)
        for doc in prior_memories:
            q = doc.get("question", "")
            sql = doc.get("sql", "")
            ans = doc.get("answer", "")
            memory.add_stm(q, {"sql": sql, "rows": doc["rows"], "answer": ans})
            self.chat_history.append(Content(role="model", parts=[Part(text=f"(question): {q} - (answer): {ans}")]))
        log.info("✅ Loaded %d memory entries into STM and chat_history", len(prior_memories))
        # ───── Step 2: Generate schema summary ─────
        schema = db_schema()
        schema_txt = "\n".join(f"{t}({', '.join(cols)})" for t, cols in schema.items())
        intro = f"Database schema:\n{schema_txt}\n\nSummarize each table and typical queries."
        summary = await asyncio.to_thread(self._llm, intro)
        memory.add_ltm_entry("__SCHEMA_SUMMARY__", "", [], summary)
        self.chat_history.append(Content(role="model", parts=[Part(text=f"(schema): {summary}")]))
        log.info(f"🧠 Added schema summary to LTM + STM \n __SCHEMA_SUMMARY__ \n{schema_txt}")
        # ───── Step 3: CoT-based self-play QA generation ─────
        for i in range(1, rounds + 1):
            log.info(f"[ColdStart-Round {i}] Generating CoT QA pairs…")
            cot_prompt = (
                "Generate 5 realistic business questions about sales data, covering trends, regions, products, customers.\n"
                "Return ONLY a JSON array with exact structure:\n"
                "[{\"question\": \"...\", \"sql\": \"SELECT ...;\"}, ...]\n"
                "Use MySQL. Each SQL must end with a semicolon."
            )
            try:
                cot_raw = await asyncio.to_thread(self._llm, cot_prompt)
                try:
                    qa_pairs = json.loads(cot_raw)
                except Exception:
                    log.warning(f"[COT-{i}] Invalid JSON returned, attempting regex fallback")
                    qa_pairs = self._parse_cot_fallback(cot_raw)
                log.info(f"CoT-{i}: Parsed {len(qa_pairs)} question–SQL candidates")
            except Exception as e:
                log.error(f"[COT-{i}] Failed to generate questions: {e}")
                continue
            # CoT loops
            for pair in qa_pairs:
                q = pair.get("question", "").strip()
                raw_sql = pair.get("sql", "").strip()
                if not q or not raw_sql:
                    continue
                try:
                    # ───── Step 3.1: Refine via Vanna ─────
                    few_shots = vanna.similar_qa(q)
                    prompt = vanna.get_sql_prompt(question=q, shots=few_shots)
                    # Ask bot agent to generate N variations 3-5 per prompt
                    sql_candidates = []
                    for _ in range(3):
                        raw = vanna.submit_prompt(prompt)
                        sql = vanna.extract_sql(raw)
                        if sql and sql.lower().startswith("select"):
                            sql_candidates.append(sql)
                    # Optional: add the bot Gemini version if valid
                    if raw_sql.lower().startswith("select"):
                        sql_candidates.append(raw_sql)
                    # ───── Step 3.2: Score + execute ─────
                    best_sql, rows = run_and_score(q, sql_candidates)
                    if not rows:
                        raise ValueError("Query returned 0 rows")
                    # ───── Step 3.3: Reason with Qwen ─────
                    reasoning_prompt = (
                        f"Q: {q}\nSample result:\n{str(rows[:5])}\n"
                        "What insight can you infer from this result?"
                    )
                    rationale = await asyncio.to_thread(self._llm, reasoning_prompt)
                    # ───── Step 3.4: Persist only valid results ─────
                    payload = {"sql": best_sql, "rows": rows, "answer": rationale}
                    memory.add_stm(q, payload)
                    memory.add_ltm_entry(q, best_sql, rows, rationale)
                    self.chat_history.append(Content(role="model", parts=[Part(text=f"(question): {q} - (answer): {rationale}")]))
                    log.info(f"✅ [COT-{i}] Stored: {q[:60]}")
                except Exception as e:
                    log.warning(f"❌ [COT-{i}] Failed to process Q: {q[:40]} — {e}")
        log.info(f"[Cold Start] completed: {rounds} CoT rounds finished")


    # ──────────── Helper: ask Gemini-Vanna for 1 SQL ────────────
    @retry_with_backoff(retries=3, delay=1.5)
    def _vanna_sql(self, question_en: str) -> str:
        few_shots = vanna.similar_qa(question_en, k=3)
        prompt = vanna.get_sql_prompt(question=question_en, shots=few_shots)
        raw    = vanna.submit_prompt(prompt)
        sql    = vanna.extract_sql(raw)
        vanna.add_question_sql(question_en, sql)
        return sql

    # ──────────── Helper: ask Qwen to brainstorm N SQLs ────────────
    async def _brainstorm_sqls(self, question_en: str, n: int = 6):
        prompt = (
            f"Give {n} distinct SQL queries that could answer the question:\n"
            f"\"{question_en}\"\nOnly return the SQL, no explanation."
        )
        raw = await asyncio.to_thread(self._llm, prompt)
        sqls = re.findall(r"SELECT .*?;", raw, flags=re.I | re.S)
        if not sqls:
            sqls = [raw.strip()] if "select" in raw.lower() else []
        return sqls


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
        attempted_sql, eval_budget = set(), 30   # Refine duplicated SQLs and setting max budget
        for attempt in range(1, max_loops + 1):
            cand_sqls = []
            # (a) Ask Vanna
            try:
                cand_sqls.append(self._vanna_sql(question_en))
            except Exception as e:
                log.warning("Vanna failed: %s", e)
            # (b) Brain-storm with Qwen (and optionally feed previous error)
            if last_error:
                self.chat_history.append(Content(role="model", parts=[Part(text=f"(error): {last_error}")]))
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
