# RecSys-Lite Docker Setup

This directory contains Docker configurations for running RecSys-Lite in various environments.

## Quick Start

To start the entire system (API and update worker):

```bash
# Development environment
docker compose -f docker/docker-compose.yml up -d

# Production environment
docker compose -f docker/docker-compose.prod.yml up -d
```

## Configuration Files

- `Dockerfile`: Main Dockerfile for the API service (optimized, <1GB size)
- `Dockerfile.worker`: Dockerfile for the update worker service
- `Dockerfile.test`: Dockerfile for running tests and development
- `docker-compose.yml`: Development docker-compose for local development
- `docker-compose.prod.yml`: Production docker-compose with resource limits and named volumes
- `docker-compose.test.yml`: Test docker-compose for running tests and CI/CD

## Docker Image Sizes

The Docker images are optimized to be under 1GB as required:

- `recsys-lite-api`: ~750MB (multi-stage build with distroless final image)
- `recsys-lite-worker`: ~700MB (optimized Python image)

## Environment Variables

### API Service

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_PATH` | Path to DuckDB database | `/data/recsys.db` |
| `MODEL_PATH` | Path to model artifacts | `/app/model_artifacts` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_WORKERS` | Number of Uvicorn workers | `4` |
| `TIMEOUT` | Request timeout in seconds | `300` |

### Update Worker

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_PATH` | Path to DuckDB database | `/data/recsys.db` |
| `MODEL_PATH` | Path to model artifacts | `/app/model_artifacts/als` |
| `UPDATE_INTERVAL` | Update interval in seconds | `60` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Volume Mounts

### Development

In development mode, local directories are mounted:

- `../data:/data`: DuckDB database and data files
- `../model_artifacts:/app/model_artifacts`: Model artifacts
- `../data/incremental:/data/incremental`: Incremental data files

### Production

In production mode, named volumes are used:

- `recsys-data`: DuckDB database and data files
- `recsys-models`: Model artifacts

## Health Checks

The API service includes a health check endpoint at `/health`. Docker is configured to use this endpoint for container health monitoring.

## Resource Limits

The production configuration includes resource limits:

- API Service: 4 CPUs, 4GB RAM
- Update Worker: 2 CPUs, 2GB RAM

## Building and Testing

```bash
# Build and test the images
docker compose -f docker/docker-compose.test.yml up

# Check image size
docker images | grep recsys-lite
```

## Troubleshooting

If you encounter issues:

1. Check container logs:
   ```bash
   docker logs recsys-lite-api
   docker logs recsys-lite-worker
   ```

2. Check container health:
   ```bash
   docker ps
   ```

3. Access API directly:
   ```bash
   curl http://localhost:8000/health
   ```

4. Check database file:
   ```bash
   docker exec -it recsys-lite-api ls -la /data
   ```