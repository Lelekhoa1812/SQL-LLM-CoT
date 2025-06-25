import os
import torch
import logging
from transformers import AutoModelForSequenceClassification
import numpy as np

logger = logging.getLogger("sql-jina")
logger.info("🚀 Starting SQL Reranker...")

class SQLReranker:
    def __init__(self):
        model_name = 'jinaai/jina-reranker-v2-base-multilingual'
        revision = '8469b0a' # safe commit hash
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype="auto",
            trust_remote_code=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def rerank(self, query: str, candidates: list):
        pairs = [[query, sql] for sql in candidates]
        scores = self.model.compute_score(pairs, max_length=512)
        best_idx = int(np.argmax(scores))
        return candidates[best_idx]