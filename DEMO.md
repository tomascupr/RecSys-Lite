# RecSys-Lite Demo

This is a simplified demo of the RecSys-Lite system, a lightweight recommendation engine designed for small e-commerce shops running on CPU-only environments.

## Demo Setup

1. Generate sample data:
   ```bash
   python data/sample_data/create_sample_data.py
   ```

2. Ingest data into DuckDB:
   ```bash
   python simple_ingest.py
   ```

3. Start the FastAPI server:
   ```bash
   python simple_api.py
   ```

4. In another terminal, test the API:
   ```bash
   python test_api.py
   ```

## Features Demonstrated

- Data ingestion into DuckDB
- Simple recommendation API with FastAPI
- Basic project structure

## Next Steps

The full system will include:
- Multiple recommendation algorithms (ALS, BPR, item2vec, LightFM)
- Hyperparameter optimization with Optuna
- Fast similarity search with Faiss
- Incremental model updates
- React widget for displaying recommendations
- Complete Docker setup for deployment

## Architecture Overview

See `docs/architecture.md` for a complete overview of the system.