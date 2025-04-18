# RecSys-Lite

Lightweight recommendation system for small e-commerce shops running on CPU-only environments.

## Features

- Supports multiple recommendation algorithms (ALS, BPR, item2vec, LightFM)
- Hyperparameter optimization with Optuna
- Fast recommendations via Faiss index
- Incremental model updates
- GDPR-compliant (EU data residency)
- FastAPI endpoint for recommendations
- React widget for displaying recommendations

## Quickstart

### Installation

```bash
# Install dependencies
poetry install

# Or use Docker
docker compose -f docker/docker-compose.yml up
```

### Data Ingestion

```bash
# Ingest data into DuckDB
recsys-lite ingest --events path/to/events.parquet --items path/to/items.csv --db recsys.db
```

### Training Models

```bash
# Train ALS model
recsys-lite train als --db recsys.db --output model_artifacts/als

# Train item2vec model
recsys-lite train item2vec --db recsys.db --output model_artifacts/item2vec

# Run hyperparameter optimization
recsys-lite optimize als --db recsys.db --trials 20
```

### API Service

```bash
# Start API service
recsys-lite serve --model model_artifacts/als --port 8000
```

### Test Recommendations

```bash
# Get recommendations for a user
curl "http://localhost:8000/recommend?user_id=U_01&k=5"
```

## Documentation

See the [docs](./docs) directory for:
- Operations runbook
- Architecture diagrams
- GDPR compliance guide
- API documentation

## Requirements

- Python 3.11+
- DuckDB 0.8.1+
- CPU with 8+ cores
- 16GB+ RAM

## License

MIT