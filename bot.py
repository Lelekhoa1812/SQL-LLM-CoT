# bot.py
import os, re, asyncio, logging, json
from utils   import db_schema
from sql     import run_and_score, vanna # vanna = GeminiVanna instance
import translation as tr
import memory
from memory import (
    add_sql_pair, retrieve_sql,
    save_table_context, get_table_context
)
import difflib
from google import genai   
from google.genai.types import Content, Part
from llm_ut  import retry_with_backoff 
from rotator import RotatingGeminiClient

log = logging.getLogger("qwen-bot")
log.info("🚀 Booting Qwen assistant")

SYSTEM_PROMPT = (
    "You are a retail-data analysis assistant. "
    "Think step-by-step, proficient in SQL, "
    "NEVER reveal your thoughts — only the final answer."
)

# ──────────── Gemini Config ────────────
genai_client = RotatingGeminiClient()  # Switch between Gemini clients when one not available
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
            rsp = genai_client.generate_content(
                model=GEMINI_MODEL,
                contents=contents
                ,stream=True
            )
            final_rsp = "".join([chunk.text for chunk in rsp if hasattr(chunk, "text")])
            self.chat_history.append(Content(role="model", parts=[Part(text=f"(question): {prompt} - (answer) {final_rsp}")]))
            return _clean_md(final_rsp)
        except Exception as e:
            log.error(f"[Gemini] memory LLM failed: {e}")
            raise

    @retry_with_backoff(retries=4, delay=1.5)
    def _llm_no_mem(self, prompt: str) -> str:
        """Stateless one-shot call"""
        try:
            rsp = genai_client.generate_content(
                model=GEMINI_MODEL,
                contents=[Content(role="user", parts=[Part(text=prompt)])]
                ,stream=True
            )
            final_rsp = "".join([chunk.text for chunk in rsp if hasattr(chunk, "text")])
            return _clean_md(final_rsp)
        except Exception as e:
            log.error(f"[Gemini] no-mem LLM failed: {e}")
            raise

    def reset_history(self):
        self.chat_history = []
        logging.info("🧼 Cleared chat_history")

    @staticmethod
    def _parse_cot_fallback(text: str) -> list[dict]:
        questions = re.findall(r"(?i)question:\s*(.+?)(?:\n|$)", text)
        sqls = re.findall(r"```sql\s*(.*?)```|(?:SELECT[\s\S]+?;)", text, flags=re.I)
        clean_sqls = [s if isinstance(s, str) else s[0] for s in sqls]
        return [{"question": q.strip(), "sql": s.strip()} for q, s in zip(questions, clean_sqls)]
    
    @staticmethod
    def is_similar_sql(new_sql, existing_list):
        return any(difflib.SequenceMatcher(None, new_sql, old).ratio() > 0.97 for old in existing_list)

    async def _schema_reason(self, text: str) -> list[str]:
        """
        Ask Gemini which table(s) the text relates to.
        Returns list of table names in lower-case.
        """
        candidate_tables = list(db_schema().keys())  # dynamic!
        tbl_contexts = [
            f"{tbl}: {get_table_context(tbl)}"
            for tbl in candidate_tables if get_table_context(tbl)
        ]
        # tbl_str = ", ".join(candidate_tables)
        # prompt = (
        #     f"Given the following user text or SQL, pick which tables "
        #     f"it is MOST related to from this list:\n{tbl_str}\n\n"
        #     f"TEXT:\n{text}\n\nReturn a JSON array of table names."
        # )
        prompt = (
            "Below are table summaries:\n"
            + "\n".join(tbl_contexts[:20]) + "\n\n"
            f"Given the following user question or SQL fragment:\n{text}\n\n"
            "Which tables are most relevant? Return a JSON array of table names."
        )
        raw = await asyncio.to_thread(self._llm_no_mem, prompt)
        try:
            tables = json.loads(_clean_md(raw))
            return [t.lower() for t in tables if t.lower() in candidate_tables]
        except Exception:
            # fallback: dumb regex
            return [t for t in candidate_tables if t.lower() in text.lower()]
    
    @retry_with_backoff(retries=3, delay=1)
    async def _generate_schema_summary(self) -> str:
        schema = db_schema()
        tables = list(schema.keys())
        summary_chunks = []
        # Breakdown to smaller tables for better accuracy
        for t in tables:
            cols = schema[t]
            coltxt = ", ".join(cols[:30]) + ("..." if len(cols) > 30 else "")
            prompt = (
                f"You're given a MySQL table named `{t}` with columns:\n{coltxt}\n\n"
                "Describe in detail:\n"
                "- What this table stores in business terms.\n"
                "- What kind of sales analysis or reporting this table enables.\n"
                "- Provide 5–10 example analytical questions someone might ask about this table.\n"
                "Return JSON like:\n"
                "{\"table\": \"...\", \"description\": \"...\", \"example_questions\": [\"...\", \"...\"]}"
            )
            try:
                res = await asyncio.to_thread(self._llm, prompt)
                cleaned = _clean_md(res)
                if not cleaned or len(cleaned) < 20:
                    raise ValueError("Empty or invalid summary")
                summary_chunks.append(cleaned)
                # Save per-chunk table vector summarization content
                save_table_context(t, cleaned)
                log.info(f"[SCHEMA] ✅ Added per-table collection \n __COLLECTION_SUMMARY__ \n{cleaned}")
            except Exception as e:
                log.warning(f"[SCHEMA] ❌ Failed for `{t}`: {e}")
        # Assemble final full schema summary
        schema_summary = "[" + ",\n".join(summary_chunks) + "]"
        return schema_summary

    @retry_with_backoff(retries=5, delay=1)
    async def _cold_start(self, rounds: int = 7):
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
        attempted_sql, eval_budget = set(), 30 # Refine duplicated SQLs and setting max budget allowance
        log.info("[Cold Start] started")
        # ───── Step 1: Load existing memory into STM/chat_history ─────
        for doc in retrieve_sql("", k=30):
            q, sql, a = doc["question"], doc["sql"], doc["answer"]
            memory.add_stm(q, {"sql": sql, "rows": doc["rows"], "answer": a})
            self.chat_history.append(Content(role="model", parts=[Part(text=f"(Q): {q} (A): {a}")]))
        log.info("✅ Loaded memory entries into STM and chat_history")
        # ───── Step 2: Generate schema summary ─────
        try:
            schema = db_schema()
            schema_txt = "\n".join(f"{t}({', '.join(cols)})" for t, cols in schema.items())
            log.info(f"[SCHEMA]\n __SCHEMA_DETAILS__ \n{schema_txt}")
            schema_summary = await self._generate_schema_summary()
            save_table_context("schema", schema_summary)
            log.info(f"[SCHEMA] Added schema summary \n __SCHEMA_SUMMARY__ \n{schema_summary}")
            self.chat_history.append(Content(role="model", parts=[Part(text=f"(schema summary):\n{schema_summary}")]))
        except Exception as e:
            log.error(f"[SCHEMA] Failed to generate schema summary: {e}")
        # ───── Step 3: CoT-based table-wise self-play QA generation ─────
        schema = db_schema(); tables = list(schema.keys()); eval_budget = set(), 500 # total budget avoid bloating
        # Gen QA for each table
        for t_idx, tbl in enumerate(tables):
            if t_idx >= 15: break  # limit max col-per-tables processed during cold-start
            colnames = schema[tbl]
            colstr = ", ".join(colnames)
            log.info(f"[ColdStart] Generating QA for table: {tbl}")
            attempted_sql = set(); tbl_budget = 50
            valid_qa_pairs = [] # collect only validated entries
            # Generic prompt
            cot_prompt = (
                f"The table `{tbl}` has the columns: {colstr}.\n"
                f"Generate 10 realistic business questions about this table ONLY.\n"
                f"Return ONLY a JSON array with this structure:\n"
                f"[{{\"question\": \"...\", \"sql\": \"SELECT ... FROM {tbl} WHERE ...;\"}}, ...]\n"
                f"Each SQL must target `{tbl}` and end with a semicolon. Use MySQL syntax."
            )
            for i in range(rounds):  # <= NEW OUTER LOOP: multiple CoT rounds per table
                log.info(f"[{tbl}] CoT-Round {i+1}/{rounds}")
                try:
                    cot_raw = await asyncio.to_thread(self._llm, cot_prompt)
                    try:
                        qa_pairs = json.loads(_clean_md(cot_raw))
                        if not qa_pairs or len(qa_pairs) == 0:
                            tbl_budget -= 5  # penalize empty round to break out
                            continue
                    except Exception:
                        log.warning(f"[{tbl}] Invalid JSON returned, attempting regex fallback")
                        qa_pairs = self._parse_cot_fallback(cot_raw)
                    log.info(f"[{tbl}] Parsed {len(qa_pairs)} question–SQL candidates")
                except Exception as e:
                    log.error(f"[{tbl}] Failed to generate questions: {e}")
                    continue
                # Grouping QA to pairs
                for pair in qa_pairs:
                    q = pair.get("question", "").strip()
                    raw_sql = pair.get("sql", "").strip()
                    if not q or not raw_sql or tbl.lower() not in raw_sql.lower():
                        continue
                    try:
                        # Step 3.1: Refine SQLs
                        few_shots = vanna.similar_qa(q, k=3)
                        prompt = vanna.get_sql_prompt(question=q, shots=few_shots)
                        sql_candidates = []
                        for _ in range(3):
                            raw = vanna.submit_prompt(prompt)
                            sql = vanna.extract_sql(raw)
                            if sql and tbl.lower() in sql.lower():
                                sql_candidates.append(sql)
                        if raw_sql.lower().startswith("select") and tbl.lower() in raw_sql.lower():
                            sql_candidates.append(raw_sql)
                        # Rm duplicated SQLs
                        dedup = []
                        for s in sql_candidates:
                            norm = re.sub(r"\s+", " ", s.lower().strip())
                            if norm not in attempted_sql and not self.is_similar_sql(norm, attempted_sql):
                                attempted_sql.add(norm)
                                dedup.append(s)
                        sql_candidates = dedup[: min(10, eval_budget, tbl_budget)]
                        num_used = len(sql_candidates)
                        eval_budget -= num_used
                        local_budget -= num_used
                        if not sql_candidates or eval_budget <= 0 or local_budget <= 0:
                            break  # break out of the QA loop for this table (keep limit to ensure healthy server)
                        if not dedup:
                            continue
                        # Step 3.2: Score + execute
                        for sql in dedup:
                            try:
                                best_sql, rows = await run_and_score(q, sql)
                                if not rows:
                                    raise ValueError("Query returned 0 rows")
                                # Step 3.3: Reasoning insight
                                rationale_prompt = (
                                    f"Q: {q}\nSample row from `{tbl}`:\n{str(rows[:5])}\n"
                                    "→ What business insight can you infer?"
                                )
                                rationale = await asyncio.to_thread(self._llm, rationale_prompt)
                                valid_qa_pairs.append((q, best_sql, rows, rationale))
                                log.info(f"[Valid SQL] {q[:60]} → {best_sql[:50]}")
                            except Exception as e:
                                log.warning(f"❌ [Invalid SQL] {sql[:60]} — {e}")
                    except Exception as e:
                        log.warning(f"❌ [{tbl}] Failed to process Q: {q[:60]} — {e}")
            # Step 3.5: Store only valid SQLs (for this table) ───
            for q, sql, rows, rationale in valid_qa_pairs:
                payload = {"sql": sql, "rows": rows, "answer": rationale}
                memory.add_stm(q, payload)
                add_sql_pair(q, sql, rows, rationale, collection_id=tbl)
                self.chat_history.append(Content(role="model", parts=[Part(text=f"(question): {q} - (answer): {rationale}")]))
            log.info(f"🎯 [{tbl}] {len(valid_qa_pairs)} valid QA pairs saved to LTM")


    # ──────────── Helper: ask Gemini-Vanna for 1 SQL ────────────
    @retry_with_backoff(retries=3, delay=1.5)
    def _vanna_sql(self, question_en: str) -> str:
        few_shots = vanna.similar_qa(question_en, k=3)
        prompt = (
            "Think step-by-step to understand the user's goal.\n"
            "Then write a safe, relevant SQL query.\n\n"
            + vanna.get_sql_prompt(question=question_en, shots=few_shots)
        )
        raw    = vanna.submit_prompt(prompt)
        sql    = vanna.extract_sql(raw)
        vanna.add_question_sql(question_en, sql)
        return sql

    # ──────────── Helper: ask LLM to brainstorm N SQLs ────────────
    async def _brainstorm_sqls(self, question_en: str, n: int = 6):
        relevant_tables = await self._schema_reason(question_en)
        schema_ctx = "\n".join([f"{t} → {get_table_context(t)}" for t in relevant_tables])
        prompt = (
            f"You are a SQL assistant. Below is relevant schema:\n{schema_ctx}\n\n"
            f"Now give {n} SQL queries to answer:\n\"{question_en}\"\n"
            "Only return raw SQL queries, no explanation."
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
        if (ltm := retrieve_sql(question_en, k=1)):
            doc = ltm[0]; memory.add_stm(question_en, doc)
            log.info("[LTM] hit")
            return doc["sql"], doc["rows"], doc["answer"]
        # Stack candidates
        attempted_sql, eval_budget = set(), 30 # Refine duplicated SQLs and setting max budget allowance
        last_error = ""
        for attempt in range(1, max_loops + 1):
            cand_sqls = []
            # (a) Ask Vanna
            try: cand_sqls.append(self._vanna_sql(question_en))
            except Exception as e: log.warning("Vanna failed: %s", e)
            # (b) Brain-storm with Qwen (and optionally feed previous error)
            if last_error:
                self.chat_history.append(Content(role="model", parts=[Part(text=f"(error): {last_error}")]))
            cand_sqls += await self._brainstorm_sqls(question_en)
            # (c) Dedup and track
            dedup = []
            for s in cand_sqls:
                norm = re.sub(r"\s+", " ", s.lower().strip())
                if norm not in attempted_sql and not self.is_similar_sql(norm, attempted_sql):
                    attempted_sql.add(norm)
                    dedup.append(s)
            cand_sqls = dedup[: min(10, eval_budget)]
            eval_budget -= len(cand_sqls)
            if not cand_sqls or eval_budget <= 0:
                break
            # Log Vanna confidence scores for debugging
            for sql in cand_sqls:
                try:
                    score = vanna.score_sql(question_en, sql)
                    log.debug(f"[VannaScore] SQL: {sql[:50]}... → Score: {score:.4f}")
                except Exception as e:
                    log.debug(f"[VannaScore] Failed to score SQL: {e}")
             # (e) Try best via run_and_score
            try:
                best_sql, rows = await run_and_score(question_en, cand_sqls)
                if not rows: raise ValueError("0 rows returned")
                # (f) Final answer generation with context-aware prompt
                tables = await self._schema_reason(best_sql)
                table_summaries = "\n".join(get_table_context(t) for t in tables)
                sample_rows = str(rows[:8])
                answer_prompt = (
                    f"Q: {question_en}\n"
                    f"Relevant Tables: {', '.join(tables)}\n"
                    f"Table Context:\n{table_summaries}\n"
                    f"Sample rows:\n{sample_rows}\n\n"
                    "Write a short and factual insight answer."
                )
                answer_en = await asyncio.to_thread(self._llm_no_mem, answer_prompt)
                answer = await self._craft_final_answer(answer_en, rows)
                payload = {"sql": best_sql, "rows": rows, "answer": answer}
                for tbl in await self._schema_reason(best_sql):
                    add_sql_pair(question_en, best_sql, rows, answer, collection_id=tbl)
                memory.add_stm(question_en, payload)
                log.info("Solved on attempt %d", attempt)
                return best_sql, rows, answer
            except Exception as e:
                last_error = str(e)
                log.warning("Attempt %d failed: %s", attempt, last_error)
        # Error
        raise RuntimeError("❌ Could not obtain a valid SQL after many tries.")

    # ──────────── Compose short natural-language answer ────────────
    async def _craft_final_answer(self, answer_en: str, rows):
        sample = str(rows[:8])
        prompt = (
            f"Answer: {answer_en}\n"
            f"Rows sample: {sample}\n\n"
            "Validate SQL response from sample and give a concise factual answer (one sentence)."
        )
        return await asyncio.to_thread(self._llm_no_mem, prompt)
