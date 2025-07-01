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
                # ,stream=True
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
                # ,stream=True
            )
            final_rsp = "".join([chunk.text for chunk in rsp if hasattr(chunk, "text")])
            return _clean_md(final_rsp)
        except Exception as e:
            log.error(f"[Gemini] no-mem LLM failed: {e}")
            raise
    
    @retry_with_backoff(retries=4, delay=1.5)
    def _llm_schema(self, prompt: str) -> str:
        """Use Gemini in non-streaming mode for schema summaries."""
        contents = [
            Content(role="model", parts=[Part(text=SYSTEM_PROMPT)]),
            Content(role="user", parts=[Part(text=prompt)])
        ]
        try:
            rsp = genai_client.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                # ,stream=True
            )
            # Gemini (non-stream) returns .text directly on rsp
            final_rsp = getattr(rsp, "text", "").strip()
            if not final_rsp:
                log.warning("[Gemini-Schema] Empty response from Gemini.")
                raise ValueError("LLM returned empty schema summary.")
            return _clean_md(final_rsp)
        except Exception as e:
            log.error(f"[Gemini-Schema] Failed to generate schema summary: {e}")
            raise

    # Hard reset on startup to avoid memory over-caching
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
    def is_similar_sql(sql_norm:str, tried:set[str], thresh:float=0.92)->bool:
        from difflib import SequenceMatcher
        for s in tried:
            if SequenceMatcher(None, sql_norm, s).ratio() > thresh:
                return True
        return False

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
    
    @retry_with_backoff(retries=2, delay=1)
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
                "- Provide 20 example analytical questions someone might ask about this table.\n"
                "Return JSON like:\n"
                "{\"table\": \"...\", \"description\": \"...\", \"example_questions\": [\"...\", \"...\"]}"
            )
            try:
                res = await asyncio.to_thread(self._llm_schema, prompt)
                log.info(f"[SCHEMA] ?? \n __COLLECTION_RAW__ \n{res}")
                # Clean & parse LLM output
                cleaned = json.loads(_clean_md(res))
                # Basic type and key check
                assert isinstance(cleaned, dict) and "description" in cleaned
                # Validate description quality
                desc = cleaned.get("description", "").strip()
                if not desc or len(desc) < 40:
                    raise ValueError("Empty or too short description")
                # Validate example question list
                if not isinstance(cleaned.get("example_questions", []), list) or len(cleaned["example_questions"]) < 5:
                    raise ValueError("Too few example questions")
                # Save to memory + summary
                summary_chunks.append(json.dumps(cleaned))
                save_table_context(t, json.dumps(cleaned))
                log.info(f"[SCHEMA] ✅ Added per-table collection \n __COLLECTION_SUMMARY__ \n{cleaned}")
            except Exception as e:
                log.warning(f"[SCHEMA] ❌ Failed for `{t}`: {e}")
        # Assemble final full schema summary
        schema_summary = "[" + ",\n".join(summary_chunks) + "]"
        return schema_summary

    async def sql_validator(
        self,
        question: str,
        tbl: str,
        max_retries: int = 5,
        eval_cap: int = 30,
    ) -> tuple[str, str, list[dict], str] | None:
        """
        Try repeatedly to generate, validate, and reason over a SQL command
        for the given question and table. Uses Gemini+Vanna CoT loop.
        Returns (question, sql, rows, rationale) if valid, else None.
        """
        sql_history = []
        attempted_sql = set()
        # Loop until exhaust
        for attempt in range(1, max_retries + 1):
            log.info(f"[{tbl}] ❓ Attempt {attempt} to solve: {question[:60]}")
            # Step A: Construct prompt
            if sql_history:
                # Reflect on past errors
                last_errors = "\n".join(
                    [f"{i+1}) SQL: {e['sql'][:50]}... → Error: {e['error']}" for i, e in sql_history[-3:]]
                )
                prompt_head = (
                    f"Re-infer business insight.\n"
                    f"Q: {question}\nTable: `{tbl}`\n"
                    f"Last attempts failed:\n{last_errors}\n"
                    f"Now retry with a better SQL using only `{tbl}`."
                )
            else:
                # First attempt: fresh CoT
                prompt_head = (
                    f"Q: {question}\nYou are writing a SQL query for the table `{tbl}` only.\n"
                    "Think step by step and return only relevant SQL statements."
                )
            # Step B: Prompt via Vanna shots
            few_shots = vanna.similar_qa(question, k=3)
            base_prompt = vanna.get_sql_prompt(question=question, shots=few_shots)
            full_prompt = f"{prompt_head}\n\n{base_prompt}"
            # Step C: Generate SQLs via CoT
            sql_candidates = []
            for _ in range(2):
                try:
                    raw = vanna.submit_prompt(full_prompt)
                    sql = vanna.extract_sql(raw)
                    if sql and tbl.lower() in sql.lower():
                        sql_candidates.append(sql)
                except Exception as e:
                    log.warning(f"[{tbl}] 🛑 SQL generation failed: {e}")
            # Step D: Dedup
            dedup = []
            for s in sql_candidates:
                norm = re.sub(r"\s+", " ", s.lower().strip())
                if norm not in attempted_sql:
                    attempted_sql.add(norm)
                    dedup.append(s)
            if not dedup:
                log.warning(f"[{tbl}] ❌ No new SQLs at attempt {attempt}")
                continue
            # Step E: Execute and validate
            for sql in dedup:
                try:
                    best_sql, rows = await run_and_score(question, sql)
                    if not rows:
                        raise ValueError("Query returned 0 rows")
                    # Step F: Insight generation prompt
                    if attempt == 1:
                        insight_prompt = (
                            f"Q: {question}\nSample rows from `{tbl}`:\n{rows[:5]}\n"
                            "→ What business insight can you infer?"
                        )
                    else:
                        insight_prompt = (
                            f"Re-infer business insight.\n"
                            f"Q: {question}\nSample rows from `{tbl}`:\n{rows[:5]}\n"
                            f"Prior failed attempts:\n{last_errors}\n"
                            "→ What insight does this new result support?"
                        )
                    # Reasoning
                    rationale = await asyncio.to_thread(self._llm, insight_prompt)
                    # Log success
                    log.info(f"[{tbl}] ✅ Success at attempt {attempt}: {best_sql[:60]}")
                    return question, best_sql, rows, rationale

                except Exception as e:
                    sql_history.append({"sql": sql, "error": str(e)})
                    log.warning(f"[{tbl}] ❌ Failed SQL: {sql[:50]} → {e}")

        log.warning(f"[{tbl}] ⛔ Failed after {max_retries} retries: {question}")
        return None

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
        schema = db_schema(); tables = list(schema.keys()); eval_budget = 500 # total budget avoid bloating
        # Gen QA for each table
        for t_idx, tbl in enumerate(tables):
            if t_idx >= 15: break  # limit max col-per-tables processed during cold-start
            colnames = schema[tbl]
            colstr = ", ".join(colnames)
            log.info(f"[ColdStart] Generating QA for table: {tbl}")
            tbl_budget = 50
            valid_qa_pairs = [] # collect only validated entries
            # Generic prompt
            table_ctx = get_table_context(tbl)
            table_desc = json.loads(table_ctx).get("description", "") if table_ctx else ""
            example_qs = json.loads(table_ctx).get("example_questions", []) if table_ctx else []
            cot_prompt = (
                f"The table `{tbl}` has the following business description:\n{table_desc}\n\n"
                f"Generate 10 realistic business questions that can be answered using this table only.\n"
                f"Each should involve a MySQL query targeting `{tbl}`.\n"
                f"Return a JSON array like:\n"
                f"[{{\"question\": \"...\", \"sql\": \"SELECT ... FROM {tbl} WHERE ...;\"}}, ...]\n"
            )
            for i in range(rounds):  # <= NEW OUTER LOOP: multiple CoT rounds per table
                log.info(f"[{tbl}] CoT-Round {i+1}/{rounds}")
                try:
                    cot_raw = await asyncio.to_thread(self._llm, cot_prompt)
                    try:
                        qa_pairs = json.loads(_clean_md(cot_raw))
                        if (not qa_pairs or len(qa_pairs) == 0) and example_qs:
                            log.warning(f"[{tbl}] ❗ Falling back to example_questions")
                            qa_pairs = [{"question": q, "sql": f"SELECT * FROM {tbl} LIMIT 10;"} for q in example_qs[:10]]
                            tbl_budget -= 5  # penalize empty round to break out
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
                    if not q: continue
                    result = await self.sql_validator(q, tbl)
                    if result:
                        valid_qa_pairs.append(result)
                        if len(valid_qa_pairs) >= 10:
                            log.info(f"[{tbl}] 🎯 Reached 10 valid QA, skipping remaining.")
                            break
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
                if not best_sql: continue
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
