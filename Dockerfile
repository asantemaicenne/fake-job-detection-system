# Dockerfile
# ==============================================================================
# Stage 1: Build dependency wheels
# ==============================================================================
FROM python:3.10-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# ==============================================================================
# Stage 2: Final Runtime Image
# ==============================================================================
FROM python:3.10-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install runtime shared libraries required by scikit-learn / OpenMP
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Install pre-built wheels from Stage 1
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application source code and configs
COPY ./src /app/src
COPY ./configs /app/configs

# Create a non-root system user for secure container execution
RUN addgroup --system api-group && \
    adduser --system --ingroup api-group api-user && \
    chown -R api-user:api-group /app

USER api-user

EXPOSE 8000

# Health check to ensure API responsiveness inside Docker networks
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
