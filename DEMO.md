# RecSys-Lite Demo

This is a simplified demo of the RecSys-Lite system, a lightweight recommendation engine designed for small e-commerce shops running on CPU-only environments.

## Demo Setup

### Prerequisites

- Python 3.11+
- DuckDB 0.8.1+
- Minimum 4GB RAM for the demo (16GB+ recommended for production)

### Step 1: Create Sample Data

First, generate the sample data that will be used for the demo:

```bash
# Generate synthetic user-item interaction data and item metadata
python data/sample_data/create_sample_data.py
```

This will create:
- `data/sample_data/events.parquet`: Sample user-item interactions
- `data/sample_data/items.csv`: Sample item metadata

### Step 2: Ingest Data

Next, ingest the data into DuckDB:

```bash
# Ingest the data into a new DuckDB database
python simple_ingest.py
```

This script:
1. Creates a new DuckDB database in memory
2. Loads events data from the Parquet file
3. Loads items data from the CSV file
4. Prepares the data for recommendation model training

### Step 3: Start the API Server

Launch the FastAPI server that will serve recommendations:

```bash
# Start the API server on port 8000
python simple_api.py
```

The server will:
1. Train a simple ALS (Alternating Least Squares) model
2. Start a FastAPI server on http://localhost:8000
3. Provide recommendation endpoints

### Step 4: Test the API

In another terminal, test the API to verify it's working:

```bash
# Test the recommendation API
python test_scripts/test_api_simple.py
```

This will make requests to the API and show sample recommendations.

## API Endpoints

The demo exposes two main endpoints:

### User Recommendations

```
GET /recommend?user_id=<user_id>&k=<k>
```

Returns recommendations for a specific user.

Example:
```bash
curl "http://localhost:8000/recommend?user_id=user_123&k=5"
```

Sample Response:
```json
{
  "user_id": "user_123",
  "recommendations": [
    {"item_id": "item_456", "score": 0.85},
    {"item_id": "item_789", "score": 0.72},
    {"item_id": "item_234", "score": 0.65},
    {"item_id": "item_567", "score": 0.58},
    {"item_id": "item_890", "score": 0.51}
  ]
}
```

### Similar Items

```
GET /similar-items?item_id=<item_id>&k=<k>
```

Returns items similar to a given item.

Example:
```bash
curl "http://localhost:8000/similar-items?item_id=item_456&k=3"
```

Sample Response:
```json
[
  {"item_id": "item_789", "score": 0.92},
  {"item_id": "item_234", "score": 0.88},
  {"item_id": "item_567", "score": 0.75}
]
```

## Understanding the Demo Code

The demo consists of three main components:

1. **Data Ingestion** (`simple_ingest.py`):
   - Shows how to load and prepare data for recommendation models
   - Creates the required database schema
   - Demonstrates working with Parquet and CSV files

2. **API Service** (`simple_api.py`):
   - Implements a minimal FastAPI service
   - Trains and serves an ALS recommendation model
   - Provides recommendation endpoints

3. **API Testing** (`test_scripts/test_api_simple.py`):
   - Demonstrates how to make requests to the API
   - Shows how to parse and use recommendation responses

## Next Steps

After exploring the demo, you can proceed to the full RecSys-Lite system which includes:

- Multiple recommendation algorithms (ALS, BPR, item2vec, LightFM, GRU4Rec)
- Hyperparameter optimization with Optuna
- Fast similarity search with Faiss
- Incremental model updates
- React widget for displaying recommendations
- Complete Docker setup for deployment
- GDPR compliance features

To learn more about these features, see:
- `docs/architecture.md` for the complete system architecture
- `README.md` for installation and usage instructions
- `docs/runbook.md` for operational guidance

## Troubleshooting

**API returns "No model found" error:**
- Make sure you've run `python simple_ingest.py` before starting the API

**No recommendations returned:**
- Check that the sample data was generated correctly
- Verify the user_id exists in the sample data

**ImportError when running the scripts:**
- Make sure you've installed all requirements: `pip install -e .` or `poetry install`

**Port already in use:**
- Change the port in `simple_api.py` if port 8000 is already in use