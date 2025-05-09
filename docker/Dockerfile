# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies and cleanup in one layer to reduce size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip

# Copy requirements to be installed first
COPY requirements.txt .

# Install dependencies directly from requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire context but use .dockerignore to exclude unnecessary files
COPY . .

# Install the package in development mode
RUN pip install --no-cache-dir -e .

# Remove unnecessary files to reduce image size
RUN find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.11/site-packages -name "__pycache__" -exec rm -rf {} +

# Final stage - Use slim image for minimal size
FROM python:3.11-slim

WORKDIR /app

# Install library dependencies required by packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only needed files from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/
COPY --from=builder /usr/local/bin/recsys-lite /usr/local/bin/
COPY --from=builder /app/src ./src

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create needed directories
RUN mkdir -p /app/data/incremental /app/model_artifacts

# Expose API port
EXPOSE 8000

# Run API server using unbuffered output
CMD ["uvicorn", "recsys_lite.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]