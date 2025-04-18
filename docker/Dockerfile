# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies and cleanup in one layer to reduce size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry==1.6.1 \
    && poetry config virtualenvs.create false

# Copy only poetry configuration files first to leverage caching
COPY pyproject.toml poetry.lock* ./

# Install production dependencies only
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY src ./src

# Remove unnecessary files to reduce image size
RUN find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.11/site-packages -name "__pycache__" -exec rm -rf {} +

# Final stage - Use distroless for minimal image size
FROM gcr.io/distroless/python3-debian11

WORKDIR /app

# Copy only needed files from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/
COPY --from=builder /app/src ./src

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create data directory 
# Note: We need to use a shell to create directories, which isn't available in distroless
# Instead, we'll ensure volumes are mounted to these paths from docker-compose

# Expose API port
EXPOSE 8000

# Run API server using unbuffered output
CMD ["uvicorn", "src.recsys_lite.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]