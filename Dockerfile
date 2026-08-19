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
