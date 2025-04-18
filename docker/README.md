# RecSys-Lite Docker Setup

This directory contains Docker configurations for deploying RecSys-Lite in various environments.

## Available Docker Files

- `Dockerfile`: Main application image
- `Dockerfile.test`: Image for running tests
- `Dockerfile.worker`: Image for update worker
- `docker-compose.yml`: Development setup
- `docker-compose.prod.yml`: Production setup
- `docker-compose.test.yml`: Test environment

## Quick Start

```bash
# Development environment
docker compose -f docker-compose.yml up -d

# Production environment
docker compose -f docker-compose.prod.yml up -d

# Test environment
docker compose -f docker-compose.test.yml up
```

## Environment Variables

The following environment variables can be configured:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_TYPE` | Type of model to use (als, bpr, item2vec, lightfm, gru4rec) | `als` |
| `API_PORT` | Port for the API service | `8000` |
| `WORKER_INTERVAL` | Interval for update worker in seconds | `60` |
| `LOG_LEVEL` | Logging level (INFO, DEBUG, WARNING, ERROR) | `INFO` |
| `THREADS` | Number of threads to use | `8` |
| `MAX_RECOMMENDATIONS` | Maximum number of recommendations per request | `50` |

## Building Images

```bash
# Build main image
docker build -t recsys-lite:latest -f Dockerfile .

# Build worker image
docker build -t recsys-lite-worker:latest -f Dockerfile.worker .
```

## Security

- The images use multi-stage builds to minimize size
- Images are based on distroless containers to reduce attack surface
- No unnecessary packages or services are included

## License

This project is licensed under the Apache License 2.0.