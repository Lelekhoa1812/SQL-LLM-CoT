from gradio_client import Client
import asyncio, logging, re, os
from utils import db_schema

log = logging.getLogger("qwen-bot")
log.info("🚀 Starting Qwen bot...")

_PROMPT_SYS = (
    "Bạn là trợ lý phân tích dữ liệu bán lẻ, nói tiếng Việt, "
    "thành thạo SQL MySQL, luôn suy nghĩ mạch lạc và có hệ thống (chain-of-thoughts) "
    "nhưng CHỈ hiển thị kết quả cuối cùng cho người dùng."
)

class QwenBot:
    def __init__(self):
        token = os.getenv("HF_TOKEN")
        self.client   = Client("Qwen/Qwen3-Demo", hf_token=token or None)
        self.settings = {
            "model": "qwen3-235b-a22b",
            "sys_prompt": _PROMPT_SYS,
            "thinking_budget": 64,
        }

    # ---------- helpers ----------
    def _call(self, prompt: str) -> str:
        try:
            res = self.client.predict(
                input_value=prompt,
                settings_form_value=self.settings,
                api_name="/add_message",
            )
            return res[0]
        except Exception as e:
            # Log chi tiết và gói gọn cho FastAPI
            log.error("⚠️ Gradio call failed: %s", e)
            raise RuntimeError("Qwen3 service tạm thời không phản hồi")

    # ---------- phase 1: generate SQL candidates ----------
    async def generate_sql_thoughts(self, question: str) -> tuple[str, list[str]]:
        schema_txt = "\n".join(f"{t}({', '.join(cols)})" for t, cols in db_schema().items())
        prompt = (f"Dựa trên lược đồ dưới đây, tạo TỐI ĐA 6 câu lệnh SQL (MySQL) "
                  f"để trả lời câu hỏi.\n===SCHEMA===\n{schema_txt}\n===CÂU HỎI===\n{question}")
        raw = await asyncio.to_thread(self._call, prompt)
        sqls = re.findall(r"SELECT .*?;", raw, flags=re.I | re.S)
        return raw, [s.replace('\n', ' ').strip() for s in sqls]

    # ---------- phase 2: craft natural-language answer ----------
    async def generate_answer(self, question: str, data: list[dict], thoughts: str) -> str:
        preview = str(data[:5])  # tránh gửi cả bảng lớn
        prompt = (f"Câu hỏi gốc: {question}\nKết quả truy vấn (mẫu): {preview}\n"
                  f"Hãy trả lời ngắn gọn, chính xác bằng tiếng Việt.")
        ans = await asyncio.to_thread(self._call, prompt)
        return ans

    # ---------- fallback loop ----------
    async def refine_until_valid(self, question: str, execute_fn, rerank_fn, max_loops=3):
        for attempt in range(max_loops):
            thoughts, sqls = await self.generate_sql_thoughts(question)
            best_sql = rerank_fn(question, sqls)
            try:
                result = execute_fn(best_sql)
                if result:
                    answer = await self.generate_answer(question, result, thoughts)
                    return best_sql, result, answer
            except Exception as e:
                log.warning("[Qwen] ⛔ SQL lỗi (%s) → thử lại vòng %d", e, attempt+1)
        raise RuntimeError("Không thể tạo SQL hợp lệ sau nhiều vòng.")
