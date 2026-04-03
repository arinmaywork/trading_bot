# ── SentiStack V2 — Trading Bot Image ───────────────────────────────────────
# Multi-stage build: keeps the final image lean (~800 MB vs ~2 GB).
#
# Usage:
#   docker build -t sentistack:latest .
#   docker run --env-file .env sentistack:latest

# ── Stage 1: Build / compile wheels ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some packages (xgboost, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: Runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="SentiStack V2"
LABEL description="NSE Algo Trading Bot — Zerodha KiteConnect + GRI + ML"

# Runtime system libs (libgomp for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash sentistack

WORKDIR /app

# Install pre-built wheels from Stage 1 (no compiler needed)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

# Copy bot source
COPY --chown=sentistack:sentistack . /app/

# Create logs directory owned by the app user
RUN mkdir -p /app/logs && chown -R sentistack:sentistack /app/logs

USER sentistack

# Health check: confirm the process is still running
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD pgrep -f "python.*main.py" > /dev/null || exit 1

CMD ["python", "main.py"]
