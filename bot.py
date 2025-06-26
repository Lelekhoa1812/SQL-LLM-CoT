from gradio_client import Client
import asyncio, logging, re, os
from utils import db_schema
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
        self.system_prompt = _SYSTEM_PROMPT
        asyncio.run(self._build_up())

    # One-turn QA for final answer summarization
    def _call(self, message: str) -> str:
        return self.client.predict(message=message, chat_history=[], api_name="/user")[0]

    # Reasoning with chat history
    def _generate(self, message: str) -> str:
        result, new_history = self.client.predict(
            message=message,
            chat_history=self.chat_history,
            api_name="/user"
        )
        self.chat_history.append((message, result))
        return result

    # Build-up logic (5 rounds recursive)
    async def _build_up(self, rounds: int = 5):
        schema = db_schema()
        schema_txt = "\n".join(f"{t}({', '.join(c)})" for t, c in schema.items())

        prompt = (
            "Dưới đây là cấu trúc database hệ thống doanh số:\n\n"
            f"{schema_txt}\n\n"
            "Hãy mô tả ngắn gọn bằng tiếng Việt về chức năng của từng bảng, "
            "cách liên kết giữa chúng và các loại truy vấn thường gặp."
        )
        reply = await asyncio.to_thread(self._generate, prompt)
        memory.add_ltm_entry("__SCHEMA_SUMMARY__", "", [{"summary": reply}], reply)
        log.info("[Qwen - Build-up 0] Đã lưu tổng quan schema")

        for i in range(1, rounds + 1):
            log.info(f"[Qwen - Build-up {i}] Reasoning round {i}...")
            followup = (
                "Dựa trên kiến thức đã có, hãy tự hỏi thêm các câu hỏi nâng cao hơn "
                "liên quan đến dữ liệu doanh số, các yếu tố thời gian, vùng miền, "
                "ngành hàng, sản phẩm. Viết ra các câu hỏi mới bằng tiếng Việt, "
                "sau đó tạo SQL tương ứng để chuẩn bị truy vấn (dù không cần kết quả ngay)."
            )
            follow_reply = await asyncio.to_thread(self._generate, followup)
            memory.add_ltm_entry(
                f"__BUILDUP_ROUND_{i}__", "", [{"thoughts": follow_reply}], follow_reply
            )

            questions = re.findall(r"Câu hỏi: (.+?)\n", follow_reply)
            queries = re.findall(r"SELECT .*?;", follow_reply, flags=re.I | re.S)
            for q, sql in zip(questions, queries):
                memory.add_ltm_entry(q, sql, [], "Chưa có kết quả – chỉ ghi nhận SQL")
                log.info(f"[Qwen - Build-up {i}] Saved question → SQL: {q[:40]}...")

        log.info("[Qwen - Build-up] ✅ Hoàn tất {rounds} vòng suy luận")

    # Phase 1: generate SQL thoughts
    async def generate_sql_thoughts(self, question: str):
        prompt = (
            f"Dựa trên lược đồ và kiến thức đã lưu, tạo TỐI ĐA 6 câu truy vấn SQL "
            f"để trả lời: {question}"
        )
        raw = await asyncio.to_thread(self._generate, prompt)
        sqls = re.findall(r"SELECT .*?;", raw, flags=re.I | re.S)
        return raw, [s.replace("\n", " ").strip() for s in sqls]

    # Phase 2: generate concise answer
    async def generate_answer(self, question: str, data: list[dict], thoughts: str) -> str:
        preview = str(data[:10])
        prompt = (
            f"Câu hỏi gốc: {question}\nKết quả truy vấn (mẫu): {preview}\n"
            f"Hãy trả lời ngắn gọn, chính xác bằng tiếng Việt."
        )
        return await asyncio.to_thread(self._call, prompt)

    # Phase 3: refine + caching
    async def refine_until_valid(self, question: str, exec_fn, rerank_fn, max_loops=3):
        if stm := memory.get_stm(question):
            log.info("[Qwen - STM] Trả kết quả nhanh")
            return stm["sql"], stm["rows"], stm["answer"]

        ltms = memory.retrieve_ltm(question, top_k=1)
        if ltms:
            e = ltms[0]
            memory.add_stm(question, e)
            log.info("[Qwen - LTM] Trả kết quả từ LTM ", {e["sql"]})
            return e["sql"], e["rows"], e["answer"]

        for attempt in range(1, max_loops + 1):
            thoughts, cands = await self.generate_sql_thoughts(question)
            best = rerank_fn(question, cands)
            try:
                rows = exec_fn(best)
                if rows:
                    ans = await self.generate_answer(question, rows, thoughts)
                    resp = {"sql": best, "rows": rows, "answer": ans}
                    memory.add_stm(question, resp)
                    memory.add_ltm_entry(question, best, rows, ans)
                    return best, rows, ans
            except Exception as e:
                log.warning("[Qwen] Vòng %d SQL lỗi: %s", attempt, e)
        raise RuntimeError("Không thể sinh SQL hợp lệ sau nhiều vòng.")
