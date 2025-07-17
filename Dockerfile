FROM python:3.11-bullseye

# ─── Environment Setup ─────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/sentence-transformers

WORKDIR /app

# ─── System Dependencies & Microsoft ODBC Driver 17 ─────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    apt-transport-https \
    ca-certificates \
    gcc \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    libltdl-dev \
    unixodbc \
    unixodbc-dev \
    wget \
    software-properties-common \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl -sSL -o packages-microsoft-prod.deb https://packages.microsoft.com/config/debian/11/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


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
# COPY --chown=user:user /home/khoa/.cache/huggingface /app/.cache/huggingface

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