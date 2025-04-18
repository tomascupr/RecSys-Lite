# RecSys-Lite Operations Runbook

This runbook provides detailed operational guidance for deploying, maintaining, and troubleshooting the RecSys-Lite recommendation system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Initial Setup](#initial-setup)
3. [Data Ingestion](#data-ingestion)
4. [Model Training](#model-training)
5. [API Operations](#api-operations)
6. [Update Worker](#update-worker)
7. [Performance Tuning](#performance-tuning)
8. [Monitoring](#monitoring)
9. [Backup and Recovery](#backup-and-recovery)
10. [GDPR Compliance](#gdpr-compliance)
11. [Troubleshooting](#troubleshooting)
12. [Maintenance Schedule](#maintenance-schedule)

## System Overview

RecSys-Lite is a lightweight recommendation system for small e-commerce shops. It consists of:

- **Data Storage**: DuckDB database storing user-item interactions and item metadata
- **Models**: Various recommendation algorithms (ALS, BPR, item2vec, LightFM, GRU4Rec)
- **API**: FastAPI service for recommendation requests
- **Update Worker**: Background process for incremental model updates
- **React Widget**: Frontend component for displaying recommendations

## Initial Setup

### Prerequisites

- 8+ CPU cores, 16GB+ RAM
- Docker and Docker Compose
- 10GB+ disk space
- Network connectivity for initial container pull only

### Deployment

#### Using Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/username/recsys-lite.git
   cd recsys-lite
   ```

2. Configure environment (optional):
   ```bash
   # Create .env file with configuration
   cat > .env << EOL
   MODEL_TYPE=als
   API_PORT=8000
   WORKER_INTERVAL=60
   EOL
   ```

3. Start the system:
   ```bash
   docker compose -f docker/docker-compose.yml up -d
   ```

4. Verify deployment:
   ```bash
   curl http://localhost:8000/health
   ```

#### Manual Deployment

1. Install dependencies:
   ```bash
   # Install Python dependencies
   pip install -e .
   ```

2. Create directory structure:
   ```bash
   mkdir -p data/incremental model_artifacts/{als,bpr,item2vec,lightfm}
   ```

3. Start API service:
   ```bash
   uvicorn recsys_lite.api.main:app --host 0.0.0.0 --port 8000
   ```

4. Start update worker (in a separate terminal):
   ```bash
   recsys-lite worker --model-dir model_artifacts/als --db data/recsys.db --interval 60
   ```

## Data Ingestion

RecSys-Lite expects two types of data:
1. **Events**: User-item interactions with timestamps and quantities
2. **Items**: Item metadata (categories, brands, prices, etc.)

### Data Format Requirements

#### Events (Parquet)
```
user_id: string      # Unique user identifier
item_id: string      # Unique item identifier
ts: int64            # Unix timestamp in seconds
qty: int             # Quantity (e.g., number of items purchased, viewed)
```

#### Items (CSV)
```
item_id: string      # Unique item identifier (primary key)
category: string     # Item category
brand: string        # Item brand
price: float         # Item price
img_url: string      # URL to item image
```

### Initial Data Load

Load initial data into the DuckDB database:

```bash
# From host machine
docker exec -it recsys-lite_recsys-lite_1 recsys-lite ingest \
  --events /data/events.parquet \
  --items /data/items.csv \
  --db /data/recsys.db
```

### Incremental Data Loading

For incremental data loading, place new event data in Parquet format in the `/data/incremental` directory.

The update worker will automatically detect and process these files based on their modification time.

```bash
# Manual incremental load (if needed)
docker exec -it recsys-lite_recsys-lite_1 recsys-lite ingest \
  --events /data/incremental/events_$(date +%Y%m%d).parquet \
  --db /data/recsys.db
```

### Schema Management

The database schema is automatically created during ingestion. To inspect the schema:

```bash
docker exec -it recsys-lite_recsys-lite_1 recsys-lite db-info --db /data/recsys.db
```

## Model Training

RecSys-Lite supports multiple recommendation models. Each model has specific strengths and use cases.

### Model Selection Guide

| Model | Strengths | Ideal For |
|-------|-----------|-----------|
| ALS | Fast, scalable, general-purpose | Most use cases, sparse data |
| BPR | Optimized for ranking quality | Ranked display of recommendations |
| item2vec | Captures item similarity | Similar item recommendations |
| LightFM | Handles cold start, uses item features | New items, content-rich catalogs |
| GRU4Rec | Captures sequential patterns | Session-based recommendations |

### Initial Training

Train your first model after data ingestion:

```bash
# Train ALS model
docker exec -it recsys-lite_recsys-lite_1 recsys-lite train als \
  --db /data/recsys.db \
  --output /app/model_artifacts/als

# Train item2vec model
docker exec -it recsys-lite_recsys-lite_1 recsys-lite train item2vec \
  --db /data/recsys.db \
  --output /app/model_artifacts/item2vec

# Train LightFM model (with item features)
docker exec -it recsys-lite_recsys-lite_1 recsys-lite train lightfm \
  --db /data/recsys.db \
  --output /app/model_artifacts/lightfm \
  --use-item-features

# Train GRU4Rec model (session-based)
docker exec -it recsys-lite_recsys-lite_1 recsys-lite train gru4rec \
  --db /data/recsys.db \
  --output /app/model_artifacts/gru4rec
```

### Hyperparameter Optimization

Improve model performance with hyperparameter optimization:

```bash
# Optimize ALS model (20 trials)
docker exec -it recsys-lite_recsys-lite_1 recsys-lite optimize als \
  --db /data/recsys.db \
  --trials 20 \
  --metric "ndcg@20" \
  --output /app/model_artifacts/als_optimized
```

The optimization will explore different parameter combinations:
- For ALS: factors, regularization, alpha
- For BPR: factors, learning_rate, regularization
- For item2vec: vector_size, window, min_count
- For LightFM: no_components, learning_rate, item_alpha, user_alpha
- For GRU4Rec: hidden_size, dropout, learning_rate

### Full Retraining Schedule

Schedule full retraining weekly or monthly:

```bash
# Automated retraining script
cat > retrain.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
MODEL_TYPE="als"  # or bpr, item2vec, lightfm
DB_PATH="/data/recsys.db"
OUTPUT_DIR="/app/model_artifacts/${MODEL_TYPE}_${DATE}"
CURRENT_DIR="/app/model_artifacts/${MODEL_TYPE}"

# Train new model
docker exec -it recsys-lite_recsys-lite_1 recsys-lite train $MODEL_TYPE \
  --db $DB_PATH \
  --output $OUTPUT_DIR

# If successful, replace current model
if [ $? -eq 0 ]; then
  docker exec -it recsys-lite_recsys-lite_1 cp -r $OUTPUT_DIR/* $CURRENT_DIR/
  echo "Model updated successfully"
else
  echo "Training failed, keeping current model"
fi
EOF

chmod +x retrain.sh
```

Add this script to crontab for weekly execution:
```bash
# Run every Sunday at 2 AM
0 2 * * 0 /path/to/retrain.sh >> /path/to/retrain.log 2>&1
```

## API Operations

The RecSys-Lite API provides endpoints for recommendations and monitoring.

### API Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/recommend` | GET | Get user recommendations | `user_id`, `k` (number of items) |
| `/similar-items` | GET | Get similar items | `item_id`, `k` (number of items) |
| `/health` | GET | Health check | None |
| `/metrics` | GET | Performance metrics | None |

### Example Requests

```bash
# Get recommendations for a user
curl "http://localhost:8000/recommend?user_id=user123&k=10"

# Get similar items
curl "http://localhost:8000/similar-items?item_id=item456&k=5"

# Check API health
curl "http://localhost:8000/health"

# Get performance metrics
curl "http://localhost:8000/metrics"
```

### Response Format

Recommendations are returned in JSON format with item metadata:

```json
{
  "user_id": "user123",
  "recommendations": [
    {
      "item_id": "item789",
      "score": 0.85,
      "title": "Product Name",
      "category": "Electronics",
      "brand": "BrandName",
      "price": 99.99,
      "img_url": "https://example.com/images/item789.jpg"
    },
    // Additional items...
  ]
}
```

### API Configuration

The API can be configured using environment variables:

```bash
# Update API configuration
docker exec -it recsys-lite_recsys-lite_1 bash -c 'cat > /app/.env << EOL
MODEL_DIR=/app/model_artifacts/als
DB_PATH=/data/recsys.db
LOG_LEVEL=INFO
MAX_RECOMMENDATIONS=50
CACHE_EXPIRY=300
EOL'

# Restart the service to apply changes
docker restart recsys-lite_recsys-lite_1
```

## Update Worker

The update worker keeps recommendations fresh by incrementally updating models with new data.

### Update Process

1. Worker polls for new events every `INTERVAL` seconds (default: 60)
2. New events are converted to a sparse user-item matrix
3. User factors are updated with `partial_fit_users`
4. New item vectors are added to the Faiss index

### Starting the Worker

```bash
# Start worker manually (if not using Docker Compose)
docker exec -it recsys-lite_recsys-lite_1 recsys-lite worker \
  --model-dir /app/model_artifacts/als \
  --db /data/recsys.db \
  --interval 60
```

### Monitoring Worker Status

```bash
# Check worker status
docker exec -it recsys-lite_recsys-lite_1 recsys-lite worker status

# View worker logs
docker exec -it recsys-lite_recsys-lite_1 tail -f /var/log/recsys-lite-worker.log
```

### Restarting the Worker

```bash
# Restart worker
docker exec -it recsys-lite_recsys-lite_1 recsys-lite worker restart
```

## Performance Tuning

Optimize RecSys-Lite for your specific workload:

### Memory Optimization

```bash
# Reduce memory usage for smaller deployments
docker exec -it recsys-lite_recsys-lite_1 bash -c 'cat > /app/.env << EOL
FAISS_NPROBE=5
BATCH_SIZE=500
CACHE_SIZE=100
EOL'
```

### CPU Optimization

```bash
# Optimize for multi-core systems
docker exec -it recsys-lite_recsys-lite_1 bash -c 'cat > /app/.env << EOL
THREADS=8
FAISS_USE_THREADS=1
EOL'
```

### Model Size vs. Quality

Adjust model size based on your catalog:

| Catalog Size | Recommended Factors | Memory Usage |
|--------------|---------------------|--------------|
| <10K items | 50-100 | Low |
| 10K-50K items | 100-200 | Medium |
| >50K items | 200-300 | High |

```bash
# Configure model size
docker exec -it recsys-lite_recsys-lite_1 recsys-lite train als \
  --db /data/recsys.db \
  --output /app/model_artifacts/als \
  --params '{"factors": 150}'
```

## Monitoring

### Health and Status

```bash
# Check API health
curl "http://localhost:8000/health"

# Get system metrics
curl "http://localhost:8000/metrics"
```

### Log Inspection

```bash
# View API logs
docker logs recsys-lite_recsys-lite_1

# View real-time logs
docker logs -f recsys-lite_recsys-lite_1
```

### Prometheus Integration

RecSys-Lite metrics can be scraped by Prometheus:

```yaml
scrape_configs:
  - job_name: 'recsys-lite'
    scrape_interval: 15s
    static_configs:
      - targets: ['recsys-lite:8000']
    metrics_path: '/metrics'
```

### Common Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| `uptime_seconds` | API uptime | N/A |
| `request_count` | Total requests | N/A |
| `recommendation_count` | Total recommendations served | N/A |
| `error_count` | Failed requests | 0 |
| `recommendations_per_second` | Throughput | Depends on workload |

## Backup and Recovery

### Database Backup

```bash
# Backup DuckDB database
docker exec -it recsys-lite_recsys-lite_1 bash -c 'cp /data/recsys.db /data/backups/recsys_$(date +%Y%m%d).db'
```

### Model Artifacts Backup

```bash
# Backup model artifacts
docker exec -it recsys-lite_recsys-lite_1 bash -c 'tar -czf /data/backups/models_$(date +%Y%m%d).tar.gz /app/model_artifacts'
```

### Automated Backup Script

```bash
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR="/path/to/backups/$DATE"
mkdir -p $BACKUP_DIR

# Backup database
docker exec recsys-lite_recsys-lite_1 bash -c "cp /data/recsys.db /data/backups/recsys_$DATE.db"
docker cp recsys-lite_recsys-lite_1:/data/backups/recsys_$DATE.db $BACKUP_DIR/

# Backup models
docker exec recsys-lite_recsys-lite_1 bash -c "tar -czf /data/backups/models_$DATE.tar.gz /app/model_artifacts"
docker cp recsys-lite_recsys-lite_1:/data/backups/models_$DATE.tar.gz $BACKUP_DIR/

# Clean up old backups (keep 30 days)
find /path/to/backups -type d -mtime +30 -exec rm -rf {} \;
EOF

chmod +x backup.sh
```

### Recovery Procedure

```bash
# Restore database
docker cp /path/to/backups/20230315/recsys_20230315.db recsys-lite_recsys-lite_1:/data/

# Restore models
docker cp /path/to/backups/20230315/models_20230315.tar.gz recsys-lite_recsys-lite_1:/data/
docker exec -it recsys-lite_recsys-lite_1 bash -c 'tar -xzf /data/models_20230315.tar.gz -C /'

# Restart service
docker restart recsys-lite_recsys-lite_1
```

## GDPR Compliance

RecSys-Lite includes tools for GDPR compliance.

### Data Inventory

User data is stored in:
1. DuckDB database (raw events)
2. Model embeddings (transformed data)
3. Faiss index (item vectors)

### User Data Export

Export all data associated with a user:

```bash
# Export a user's data
docker exec -it recsys-lite_recsys-lite_1 recsys-lite gdpr export-user \
  --user-id USER_ID \
  --db /data/recsys.db \
  --output /data/exports/user_USER_ID.json
```

The export contains:
- Raw event data
- Items the user interacted with
- Timestamps and quantities

### User Data Deletion

Delete a user's data from the system:

```bash
# Delete a user's data
docker exec -it recsys-lite_recsys-lite_1 recsys-lite gdpr delete-user \
  --user-id USER_ID \
  --db /data/recsys.db

# Retrain models or wait for the update worker to reflect changes
```

The deletion process:
1. Removes raw events from DuckDB
2. Marks the user for removal in models
3. Update worker applies changes to embeddings

### Data Retention Policy

Configure data retention period:

```bash
# Set 6-month retention policy
docker exec -it recsys-lite_recsys-lite_1 recsys-lite configure \
  --retention-days 180
```

### Right to be Forgotten Workflow

1. Receive deletion request
2. Execute deletion command
3. Provide confirmation with timestamp
4. Document deletion in compliance log
5. Rotate backups to ensure complete removal

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

3. Verify database connection:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 recsys-lite db-check --db /data/recsys.db
   ```

4. Check model artifacts:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 ls -la /app/model_artifacts
   ```

5. Restart service:
   ```bash
   docker restart recsys-lite_recsys-lite_1
   ```

### Poor Recommendation Quality

1. Check data quality:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 recsys-lite validate --db /data/recsys.db
   ```

2. Check model metrics:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 recsys-lite evaluate \
     --model-dir /app/model_artifacts/als \
     --db /data/recsys.db
   ```

3. Common issues and solutions:
   - **Cold start problem**: Use LightFM with item features
   - **Too generic recommendations**: Increase model factors
   - **Poor similar items**: Try item2vec or increase training data
   - **Missing recent behavior**: Check update worker logs

4. Try different model types:
   ```bash
   # Try BPR model instead of ALS
   docker exec -it recsys-lite_recsys-lite_1 recsys-lite train bpr \
     --db /data/recsys.db \
     --output /app/model_artifacts/bpr
   ```

### Update Worker Issues

1. Check worker status:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 recsys-lite worker status
   ```

2. Check worker logs:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 cat /var/log/recsys-lite-worker.log
   ```

3. Common issues:
   - **Worker not updating**: Check incremental data path
   - **Memory errors**: Reduce batch size
   - **Slow updates**: Check event volume, adjust interval

4. Restart worker:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 recsys-lite worker restart
   ```

### Database Corruption

1. Check database integrity:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 recsys-lite db-check \
     --db /data/recsys.db
   ```

2. Restore from backup if necessary:
   ```bash
   docker exec -it recsys-lite_recsys-lite_1 cp /data/backups/recsys_20230315.db /data/recsys.db
   ```

## Maintenance Schedule

Follow this recommended maintenance schedule:

### Daily
- Check API health (`curl http://localhost:8000/health`)
- Review error logs
- Verify update worker status

### Weekly
- Run full model retraining
- Backup database and models
- Check recommendation metrics
- Clean up old incremental files

### Monthly
- Run hyperparameter optimization
- Perform disk cleanup
- Review performance metrics
- Test recovery procedures

### Quarterly
- Update to latest RecSys-Lite version
- Audit GDPR compliance
- Review and update retention policies
- Benchmark system performance