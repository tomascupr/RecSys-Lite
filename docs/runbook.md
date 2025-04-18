# RecSys-Lite Operations Runbook

## System Overview

RecSys-Lite is a lightweight recommendation system for small e-commerce shops. It consists of:

- **Data Storage**: DuckDB database storing user-item interactions and item metadata
- **Models**: Various recommendation algorithms (ALS, BPR, item2vec, LightFM)
- **API**: FastAPI service for recommendation requests
- **Updater**: Background worker process for model updates

## Initial Setup

### Prerequisites

- 8+ CPU cores, 16GB+ RAM
- Docker and Docker Compose
- 10GB+ disk space

### Deployment

1. Clone the repository
2. Navigate to the project directory
3. Start the system:

```bash
docker compose -f docker/docker-compose.yml up -d
```

## Data Ingestion

### Initial Data Load

```bash
# From host machine
docker exec -it recsys-lite_recsys-lite_1 recsys-lite ingest --events /data/events.parquet --items /data/items.csv --db /data/recsys.db
```

### Incremental Data Load

For incremental data loading, place new event data in Parquet format in the `/data/incremental` directory.

```bash
# Auto-detected by update worker
docker exec -it recsys-lite_recsys-lite_1 recsys-lite ingest --events /data/incremental/events_20230315.parquet --db /data/recsys.db
```

## Model Training

### Initial Training

```bash
# Train ALS model
docker exec -it recsys-lite_recsys-lite_1 recsys-lite train als --db /data/recsys.db --output /app/model_artifacts/als

# Train item2vec model
docker exec -it recsys-lite_recsys-lite_1 recsys-lite train item2vec --db /data/recsys.db --output /app/model_artifacts/item2vec
```

### Hyperparameter Optimization

```bash
docker exec -it recsys-lite_recsys-lite_1 recsys-lite optimize als --db /data/recsys.db --trials 20 --output /app/model_artifacts/als_optimized
```

### Full Retraining

Schedule full retraining weekly or monthly:

```bash
docker exec -it recsys-lite_recsys-lite_1 recsys-lite train als --db /data/recsys.db --output /app/model_artifacts/als_new

# Once successful, replace the old model
docker exec -it recsys-lite_recsys-lite_1 cp -r /app/model_artifacts/als_new/* /app/model_artifacts/als/
```

## Monitoring

### Health Check

```bash
curl "http://localhost:8000/health"
```

### Logs

```bash
docker logs recsys-lite_recsys-lite_1
```

### Metrics

Key metrics are exposed at `/metrics` endpoint for Prometheus scraping.

## GDPR Compliance

### Data Backup

```bash
# Backup the DuckDB database
docker exec -it recsys-lite_recsys-lite_1 cp /data/recsys.db /data/backups/recsys_$(date +%Y%m%d).db
```

### User Data Deletion

```bash
# Delete a user's data
docker exec -it recsys-lite_recsys-lite_1 recsys-lite gdpr delete-user --user-id USER_ID --db /data/recsys.db

# Retrain models after user deletion
docker exec -it recsys-lite_recsys-lite_1 recsys-lite train als --db /data/recsys.db --output /app/model_artifacts/als
```

### Data Export

```bash
# Export a user's data
docker exec -it recsys-lite_recsys-lite_1 recsys-lite gdpr export-user --user-id USER_ID --db /data/recsys.db --output /data/exports/user_USER_ID.json
```

## Troubleshooting

### API Not Responding

1. Check container status:
   ```bash
   docker ps | grep recsys-lite
   ```

2. Check logs:
   ```bash
   docker logs recsys-lite_recsys-lite_1
   ```

3. Restart service:
   ```bash
   docker restart recsys-lite_recsys-lite_1
   ```

### Poor Recommendation Quality

1. Check data quality:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 recsys-lite validate --db /data/recsys.db
   ```

2. Retrain models with more data or different parameters.

### Update Worker Issues

1. Check worker status:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 recsys-lite worker status
   ```

2. Restart worker:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 recsys-lite worker restart
   ```