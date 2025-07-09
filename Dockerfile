FROM python:3.11-slim

# ─── Environment Setup ─────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/sentence-transformers

WORKDIR /app

# ─── System Dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    ca-certificates \
    apt-transport-https && \
    curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.gpg && \
    curl -sSL https://packages.microsoft.com/config/debian/11/prod.list -o /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    apt-get remove -y libodbc2 libodbcinst2 unixodbc-common && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
    msodbcsql17 \
    unixodbc \
    unixodbc-dev \
    gcc \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    libltdl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ─── Python Dependencies ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ─── Create non-root user ──────────────────────────────────────
RUN useradd -m -u 1000 user

# Create all needed cache paths and fix permissions
RUN mkdir -p /app \
    && mkdir -p /app/.cache/huggingface/hub \
    && mkdir -p /app/model_cache \
    && mkdir -p /app/history \
    && chown -R user:user /app

# ─── Preload model ─────────────────────────────────────────────
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ─── Copy App ──────────────────────────────────────────────────
COPY . .

# ─── HF Model cache ────────────────────────────────────────────
RUN mkdir -p /tmp/hf_cache && chown -R user:user /tmp/hf_cache

# ─── Switch to non-root user ───────────────────────────────────
USER user

# ─── Run the API ───────────────────────────────────────────────
CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860"]
