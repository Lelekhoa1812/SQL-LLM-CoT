# Dockerfile
FROM python:3.12-slim

# ─── Environment Setup ─────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/sentence-transformers

WORKDIR /app

# ─── System Dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# ─── Python Dependencies ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 user

# Create all needed cache paths and fix permissions for non-root user
RUN mkdir -p /app \
    && mkdir -p /app/.cache/huggingface/hub \
    && mkdir -p /app/model_cache \
    && chown -R user:user /app

# ─── Model preloader ───────────────────────────────────────────
RUN python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('jinaai/jina-reranker-v2-base-multilingual', revision='8469b0a', trust_remote_code=True)"
# RUN python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B', trust_remote_code=True)"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy project files
COPY . .

# Switch to non-root user (important AFTER chown)
USER user

# ─── Run Server ────────────────────────────────────────────────
CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860"]
