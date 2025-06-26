from gradio_client import Client
import asyncio, logging, re, os
from utils import db_schema, execute_sql
from sql   import SQLReranker
import memory

log = logging.getLogger("qwen-bot")
log.info("🚀 Starting Qwen bot...")

_SYSTEM_PROMPT = (
    "Bạn là trợ lý phân tích dữ liệu bán lẻ, nói tiếng Việt, "
    "thành thạo SQL, suy nghĩ mạch lạc và có hệ thống "
    "CHỈ hiển thị kết quả cuối cùng cho người dùng."
)

class QwenBot:
    def __init__(self):
        token = os.getenv("HF_TOKEN")
        self.client = Client("mikeee/qwen-7b-chat", hf_token=token or None)
        self.chat_history: list[tuple[str, str]] = []  # STM-style memory
        self.reranker = SQLReranker()
        self.system_prompt = _SYSTEM_PROMPT
        asyncio.run(self._build_up())

    def _call(self, message: str) -> str:
            """
            One‐off call (no memory) for summarization/final answer.
            """
            ai_reply, _ = self.client.predict(
                message=message,
                chat_history=[],
                api_name="/user"
            )
            return ai_reply
    
    def _generate(self, message: str) -> str:
        """
        Single turn with conversation memory.
        Sends (message, chat_history) → returns ai_reply and updates chat_history.
        """
        ai_reply, new_history = self.client.predict(
            message=message,
            chat_history=self.chat_history,
            api_name="/user"
        )
        self.chat_history = new_history
        return ai_reply

    # Build-up logic (10 rounds recursive)
    async def _build_up(self, rounds: int = 10):
        """
        1) Load existing LTM entries and seed STM (chat_history)
        2) Describe schema
        3) Repeat CoT rounds: propose Q→SQL, rerank, exec, reason, save LTM
        """
        # 1) seed from LTM
        existing = memory.retrieve_ltm("", top_k=100)
        for e in existing:
            summary = f"Tôi nhớ: '{e['question']}' → SQL: {e['sql']}"
            self.chat_history.append((summary, e['answer'] or ""))
            log.info(f"[Qwen - Seed] Memory: '{e['question']}' → SQL: {e['sql']}")
        
        # 2) describe schema
        schema = db_schema()
        schema_txt = "\n".join(f"{t}({', '.join(cols)})" for t, cols in schema.items())
        # Prompt engineering
        intro = (
            f"Cấu trúc database:\n{schema_txt}\n\n"
            "Hãy mô tả ngắn gọn bằng tiếng Việt về chức năng từng bảng, "
            "cách liên kết và các loại truy vấn hay dùng."
        )
        summary = await asyncio.to_thread(self._generate, intro)
        memory.add_ltm_entry("__SCHEMA_SUMMARY__", "", [], summary)
        log.info(f"[Qwen - Build-up 0] Saved schema summary {summary}")
        
        # 3) CoT enrichment rounds
        for i in range(1, rounds + 1):
            log.info(f"[Build-up {i}] Thinking of new Q→SQL pairs…")
            cot_prompt = (
                "Dựa trên kiến thức đã có, tự hỏi thêm các câu hỏi nâng cao "
                "về doanh số (thời gian, khu vực, ngành hàng), rồi tạo SQL mẫu."
            )
            cot = await asyncio.to_thread(self._generate, cot_prompt)
            # extract questions & SQLs
            questions = re.findall(r"Câu hỏi:\s*(.+)", cot)
            sqls      = re.findall(r"(SELECT .*?;)", cot, flags=re.IGNORECASE|re.DOTALL)
            # save the entire CoT blob
            memory.add_ltm_entry(f"__BUILDUP_ROUND_{i}__", "", [], cot)
            log.info(f"[Qwen - Build-up {i}] CoT: {cot}")
            # for each Q→SQL, rerank, exec & reason
            for q, raw_sql in zip(questions, sqls):
                best_sql = self.reranker.rerank(q, [raw_sql])  # single candidate
                rows     = execute_sql(best_sql)
                # feed back to model to reason on results
                reason_prompt = (
                    f"Kết quả truy vấn:\n{rows[:5]}\n\n"
                    "Dựa vào đó, cho insight thêm hoặc phát hiện gì mới?"
                )
                reasoning = await asyncio.to_thread(self._generate, reason_prompt)
                # persist
                memory.add_ltm_entry(q, best_sql, rows, reasoning)
                log.info(f"[Qwen - Build-up {i}] Saved Q→SQL→Reason for: {q[:30]}…")
        log.info(f"[Qwen - Build-up] ✅ Hoàn tất {rounds} vòng enrichment")

    # Phase 1: generate SQL thoughts
    async def generate_sql_thoughts(self, question: str):
        """
        Generate up to 6 SQL candidates for a new user question.
        """
        prompt = (
            "Dựa trên lược đồ và kiến thức đã lưu, tạo tối đa 6 câu truy vấn SQL "
            f"để trả lời: {question}"
        )
        raw = await asyncio.to_thread(self._generate, prompt)
        sqls = re.findall(r"(SELECT .*?;)", raw, flags=re.IGNORECASE|re.DOTALL)
        log.info(f"[Qwen - SQL] {sqls}")
        return raw, [s.replace("\n"," ").strip() for s in sqls]

    # Phase 2: generate concise answer
    async def generate_answer(self, question: str, data: list[dict], thoughts: str) -> str:
        """
        Craft a concise VN answer from raw SQL results.
        """
        preview = str(data[:10])
        prompt = (
            f"Câu hỏi: {question}\n"
            f"Kết quả mẫu: {preview}\n"
            "Hãy trả lời ngắn gọn và chính xác bằng tiếng Việt."
        )
        res = await asyncio.to_thread(self._call, prompt)
        log.info(f"[Qwen - Answer] {res}")
        return res

    # Phase 3: refine + caching
    async def refine_until_valid(
        self, question: str, exec_fn, rerank_fn, max_loops: int = 3
    ):
        """
        1) STM hit?
        2) LTM hit?
        3) Fallback: loop Q→SQL→rerank→exec→answer up to max_loops
        """
        # STM
        if stm := memory.get_stm(question):
            log.info("[STM] re-using cached")
            return stm["sql"], stm["rows"], stm["answer"]
        # LTM
        ltms = memory.retrieve_ltm(question, top_k=1)
        if ltms:
            e = ltms[0]
            memory.add_stm(question, e)
            log.info("[LTM] re-using memory SQL")
            return e["sql"], e["rows"], e["answer"]
        # fallback
        for attempt in range(1, max_loops+1):
            raw, cands = await self.generate_sql_thoughts(question)
            best = rerank_fn(question, cands)
            try:
                rows = exec_fn(best)
                if rows:
                    ans = await self.generate_answer(question, rows, raw)
                    resp = {"sql": best, "rows": rows, "answer": ans}
                    memory.add_stm(question, resp)
                    memory.add_ltm_entry(question, best, rows, ans)
                    return best, rows, ans
            except Exception as e:
                log.warning(f"[Fallback {attempt}] SQL error: {e}")
        raise RuntimeError("Không thể tạo SQL hợp lệ sau nhiều vòng.")