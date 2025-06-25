import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenBot:
    def __init__(self):
        model_name = os.getenv("QWEN_MODEL", "Qwen/Qwen3-4B")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.model.eval()

    async def generate_sql_thoughts(self, question: str):
        prompt = f"Generate 3 candidate SQL queries for the question: {question}"
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=1024)[0].tolist()
        # parse thinking vs. actual content
        think_token = 151668  # </think>
        try:
            idx = len(output_ids) - output_ids[::-1].index(think_token)
        except ValueError:
            idx = 0
        thinking = self.tokenizer.decode(output_ids[:idx], skip_special_tokens=True)
        content = self.tokenizer.decode(output_ids[idx:], skip_special_tokens=True)
        # extract SQL statements
        sql_candidates = [line.strip() for line in content.split('\n') if line.strip().upper().startswith("SELECT")]
        return thinking, sql_candidates

    async def generate_answer(self, question: str, result, thinking: str):
        prompt = f"Based on the result {result}, answer the question: {question}"
        messages = [
            {"role": "system", "content": thinking},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=1024)[0]
        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return answer