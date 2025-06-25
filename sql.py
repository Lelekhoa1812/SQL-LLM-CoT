import torch, numpy as np, logging
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger("sql-jina")
logger.info("🚀 Starting SQL reranker...")

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

    def rerank(self, query: str, sqls: list[str]) -> str:
        if len(sqls) == 1:
            return sqls[0]
        pairs = [[query, s] for s in sqls]
        scores = self.model.compute_score(pairs, max_length=512)
        best = sqls[int(np.argmax(scores))]
        logger.info("[Jina]🔝 Reranker chọn: %s", best)
        return best