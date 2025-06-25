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

# ─── Create Cache Dirs & Fix Permissions ───────────────────────
RUN mkdir -p /app/model_cache /app/.cache/huggingface/sentence-transformers && \
    adduser --disabled-password --gecos "" --uid 1000 user && \
    chown -R user:user /app/model_cache /app/.cache /app

# ─── Switch to Non-root User ───────────────────────────────────
USER user

# ─── Copy Application Files ────────────────────────────────────
COPY . .

# ─── Run Server ────────────────────────────────────────────────
CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860"]
