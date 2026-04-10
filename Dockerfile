# ──────────────────────────────────────────────────────────────────────────────
# Micro-SWE Gym — Dockerfile
# Target: Python 3.10, 2 vCPU / 8 GB RAM
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Metadata
LABEL maintainer="micro-swe-gym"
LABEL description="OpenEnv environment for automated PR review/fixing"

# Prevent .pyc files and enable unbuffered stdout (important for structured logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY models.py        ./models.py
COPY inference.py     ./inference.py
COPY server/          ./server/
COPY openenv.yaml     ./openenv.yaml

# ── Health check (Change 8000 to 7860) ─────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Expose FastAPI port (Change 8000 to 7860) ──────────────────────────────────
EXPOSE 7860

# ── Default command ───────────────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port 7860 & sleep 5 && python inference.py"]
