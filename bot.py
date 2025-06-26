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
        self.client = Client("kz919/Qwen3-0.6B-Zero-GPU", hf_token=token or None)
        self.settings = {
            "system_message": _SYSTEM_PROMPT,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        # Build-up Phase: Run once on start-up
        asyncio.run(self._build_up())

    # Single prompt call to /chat
    def _generate(self, message: str, **overrides):
        data = {"message": message, **self.settings, **overrides}
        return self.client.predict(**data, api_name="/chat") 
    
    # Build up task (looping 5 times)
    async def _build_up(self, rounds: int = 5):
        schema = db_schema()
        schema_txt = "\n".join(f"{t}({', '.join(c)})" for t, c in schema.items())
        # Prompt engineering and STM caching here
        history = []
        prompt = (
            "Dưới đây là cấu trúc database hệ thống doanh số:\n\n"
            f"{schema_txt}\n\n"
            "Hãy mô tả ngắn gọn bằng tiếng Việt về chức năng của từng bảng, "
            "cách liên kết giữa chúng và các loại truy vấn thường gặp."
        )
        reply, history = await asyncio.to_thread(self._generate, [], message=prompt)
        memory.add_ltm_entry("__SCHEMA_SUMMARY__", "", [{"summary": reply}], reply)
        log.info("[Qwen - Build-up 0] Đã lưu tổng quan schema")
        # Recursive iteration
        for i in range(1, rounds + 1):
            log.info(f"[Qwen - Build-up {i}] Reasoning round {i}...")
            followup = (
                "Dựa trên kiến thức đã có, hãy tự hỏi thêm các câu hỏi nâng cao hơn "
                "liên quan đến dữ liệu doanh số, các yếu tố thời gian, vùng miền, "
                "ngành hàng, sản phẩm. Viết ra các câu hỏi mới bằng tiếng Việt, "
                "sau đó tạo SQL tương ứng để chuẩn bị truy vấn (dù không cần kết quả ngay)."
            )
            follow_reply, history = await asyncio.to_thread(
                self._generate, history, user_message=followup
            )
            # Save CoT reasoning to LTM
            questions = re.findall(r"Câu hỏi: (.+?)\n", follow_reply)
            queries = re.findall(r"SELECT .*?;", follow_reply, flags=re.I | re.S)
            memory.add_ltm_entry(
                f"__BUILDUP_ROUND_{i}__", "", [{"thoughts": follow_reply}], follow_reply
            )
            # Index each QA-SQL pair
            for q, sql in zip(questions, queries):
                memory.add_ltm_entry(q, sql, [], "Chưa có kết quả – chỉ ghi nhận SQL")
                log.info(f"[Qwen - Build-up {i}] Saved question → SQL: {q[:40]}...")
        log.info("[Qwen - Build-up] ✅ Đã hoàn tất %d vòng suy luận sâu", rounds)

    # ---------- phase 1: generate SQL candidates ----------
    async def generate_sql_thoughts(self, question: str):
        prompt = (
            f"Dựa trên lược đồ và kiến thức đã lưu, tạo TỐI ĐA 6 câu SQL MySQL "
            f"để trả lời: {question}"
        )
        raw, _ = await asyncio.to_thread(self._generate, [], user_message=prompt)
        sqls = re.findall(r"SELECT .*?;", raw, flags=re.I|re.S)
        return raw, [s.replace("\n"," ").strip() for s in sqls]

    # ---------- phase 2: craft natural-language answer ----------
    async def generate_answer(self, question: str, data: list[dict], thoughts: str) -> str:
        preview = str(data[:10])  # Avoid full large table, trim to 10 rows
        prompt = (f"Câu hỏi gốc: {question}\nKết quả truy vấn (mẫu): {preview}\n"
                  f"Hãy trả lời ngắn gọn, chính xác bằng tiếng Việt.")
        ans = await asyncio.to_thread(self._call, prompt)
        return ans

    # ---- phase 3: execution phase with fallback and memo -------
    async def refine_until_valid(self, question: str, exec_fn, rerank_fn, max_loops=3):
        # 1) STM?
        if stm := memory.get_stm(question):
            log.info("[Qwen - STM] Trả kết quả nhanh")
            return stm["sql"], stm["rows"], stm["answer"]
        # 2) LTM?
        ltms = memory.retrieve_ltm(question, top_k=1)
        if ltms:
            e = ltms[0]
            memory.add_stm(question, e)
            log.info("[Qwen - LTM] Trả kết quả từ LTM")
            return e["sql"], e["rows"], e["answer"]
        # 3) Fallback loop
        for attempt in range(1, max_loops+1):
            thoughts, cands = await self.generate_sql_thoughts(question)
            best = rerank_fn(question, cands)
            try:
                rows = exec_fn(best)
                if rows:
                    ans = await self.generate_answer(question, rows, thoughts)
                    resp = {"sql":best, "rows":rows, "answer":ans}
                    memory.add_stm(question, resp)
                    memory.add_ltm_entry(question, best, rows, ans)
                    return best, rows, ans
            except Exception as e:
                log.warning("[Qwen] Vòng %d SQL lỗi: %s", attempt, e)
        raise RuntimeError("Không thể sinh SQL hợp lệ sau nhiều vòng.")
