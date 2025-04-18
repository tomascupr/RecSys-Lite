# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install Poetry
RUN pip install poetry==1.6.1

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY src ./src

# Final stage
FROM gcr.io/distroless/python3-debian11

WORKDIR /app

# Copy installed packages and application from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/
COPY --from=builder /app/src ./src

# Create directories for data
RUN mkdir -p /data/model_artifacts

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "src.recsys_lite.api.main:app", "--host", "0.0.0.0", "--port", "8000"]