# syntax=docker/dockerfile:1
# ===========================================================================
# GastricPath AI — Production Container Definition
# Multi-stage build for lightweight, secure, non-root CPU inference
# ===========================================================================

FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and directories
RUN useradd -u 1000 -m -s /bin/bash gastricpath && \
    mkdir -p /app/uploads /app/data && \
    chown -R gastricpath:gastricpath /app

WORKDIR /app

# Install Python dependencies using CPU wheels
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application and research assets
COPY --chown=gastricpath:gastricpath app/ app/
COPY --chown=gastricpath:gastricpath models/ models/
COPY --chown=gastricpath:gastricpath research/ research/
COPY --chown=gastricpath:gastricpath docs/ docs/
COPY --chown=gastricpath:gastricpath .env.example .env

# Switch to non-root execution
USER gastricpath

# Expose HTTP port
EXPOSE 8000

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch application with production ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

