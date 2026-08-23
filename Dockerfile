FROM python:3.12-slim AS base

# Faster, quieter, more deterministic pip installs; unbuffered stdout so
# log lines show up immediately under `docker compose logs -f`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so this layer is only rebuilt when requirements.txt
# actually changes rather than on every source edit.
COPY requirements.txt .
# sentence-transformers (rag/reranker.py, EMBED_BACKEND=local) pulls in
# torch — installed here as the CPU-only wheel first so pip's resolver
# satisfies that dependency without ever considering the default
# CUDA-enabled build. The CUDA build assumes a GPU no deployment target of
# this image actually has (Render/Railway/Fly free tiers, this repo's own
# docker-compose.yml) and its bundled nvidia-*/CUDA packages roughly 10x
# the image size and memory footprint — enough to OOM-loop a 512MB-RAM
# free-tier host during model load, which is silent (no crash log, just a
# request that never completes) and easy to mistake for "still starting up."
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY . .

# Runs as a non-root user; own only what the process needs to write (the
# on-disk corpus cache from rag/ingestion.py).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# No default CMD — the API and UI are two different entry points into the
# same image, selected explicitly per-service in docker-compose.yml.
