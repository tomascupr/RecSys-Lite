# RecSys-Lite

A lightweight recommendation system designed for small e-commerce shops running on CPU-only environments. RecSys-Lite provides multiple recommendation algorithms, automatic hyperparameter optimization, fast recommendation serving via Faiss, and a React widget for displaying recommendations.

## Features

- **Multiple Recommendation Algorithms**:
  - ALS and BPR via the `implicit` library
  - Item2Vec embeddings via `gensim`
  - Hybrid matrix factorization via `LightFM`
  - Optional GRU4Rec session-based model via PyTorch

- **Hyperparameter Optimization**:
  - Automatic tuning via Optuna
  - Optimizes for HR@20 and NDCG@20 metrics
  - Saves best model parameters

- **Fast Recommendation Serving**:
  - Faiss IVF-Flat index for approximate nearest neighbor search
  - FastAPI endpoint for recommendations
  - Incremental model updates
  
- **React Widget**:
  - Responsive carousel for displaying recommendations
  - Customizable with CSS classes
  - Available as NPM package

- **GDPR Compliance**:
  - EU data residency
  - User data export and deletion functionality
  - Complete audit trail

## Architecture

The system is composed of several components:

1. **Data Layer**: DuckDB for storing user-item interactions and item metadata
2. **Model Layer**: Various recommendation algorithms and Optuna for hyperparameter optimization
3. **Serving Layer**: Faiss for ANN search, FastAPI for the recommendation API
4. **Frontend Layer**: React widget for displaying recommendations
5. **Update Worker**: Background process for incremental model updates

For a more detailed architecture overview, see [docs/architecture.md](docs/architecture.md).

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tomascupr/RecSys-Lite.git
cd RecSys-Lite

# Install Python dependencies
poetry install

# Or use Docker
docker compose -f docker/docker-compose.yml up -d
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
recsys-lite optimize als --db recsys.db --trials 20 --metric ndcg@20
```

### API Service

```bash
# Start API service
recsys-lite serve --model model_artifacts/als --port 8000
```

### Update Worker

```bash
# Start update worker for incremental model updates
recsys-lite worker --model model_artifacts/als --db recsys.db --interval 60
```

### React Widget

```bash
# Install the widget package
npm install recsys-lite-widget

# Import and use in your React application
import { RecommendationCarousel } from 'recsys-lite-widget';

function App() {
  return (
    <RecommendationCarousel
      apiUrl="https://your-api-url"
      userId="user123"
      count={5} 
    />
  );
}
```

## CLI Commands Reference

### Data Management

| Command | Description | Options |
|---------|-------------|---------|
| `ingest` | Ingest data into DuckDB | `--events <parquet>`, `--items <csv>`, `--db <path>` |
| `gdpr export-user` | Export user data | `--user-id <id>`, `--db <path>`, `--output <json>` |
| `gdpr delete-user` | Delete user data | `--user-id <id>`, `--db <path>` |

### Model Training

| Command | Description | Options |
|---------|-------------|---------|
| `train als` | Train ALS model | `--db <path>`, `--output <dir>`, `--test-size <float>` |
| `train bpr` | Train BPR model | `--db <path>`, `--output <dir>`, `--test-size <float>` |
| `train item2vec` | Train Item2Vec model | `--db <path>`, `--output <dir>`, `--test-size <float>` |
| `train lightfm` | Train LightFM model | `--db <path>`, `--output <dir>`, `--test-size <float>` |
| `train gru4rec` | Train GRU4Rec model | `--db <path>`, `--output <dir>`, `--test-size <float>` |

### Hyperparameter Optimization

| Command | Description | Options |
|---------|-------------|---------|
| `optimize` | Run hyperparameter optimization | `--model-type <type>`, `--db <path>`, `--output <dir>`, `--metric <metric>`, `--trials <int>` |

### Serving

| Command | Description | Options |
|---------|-------------|---------|
| `serve` | Start the FastAPI server | `--model-dir <path>`, `--host <host>`, `--port <port>` |
| `worker` | Start the update worker | `--model-dir <path>`, `--db <path>`, `--interval <seconds>` |

## API Endpoints

### Recommendation Endpoint

```
GET /recommend?user_id=<user_id>&k=<k>
```

Returns a list of recommended items for a user.

**Parameters**:
- `user_id` (required): User ID to get recommendations for
- `k` (optional): Number of recommendations to return (default: 10)

**Response**:
```json
{
  "user_id": "user123",
  "recommendations": [
    {
      "item_id": "item456",
      "score": 0.95,
      "title": "Product Title",
      "category": "Category",
      "brand": "Brand",
      "price": 99.99,
      "img_url": "https://example.com/image.jpg"
    },
    ...
  ]
}
```

### Similar Items Endpoint

```
GET /similar-items?item_id=<item_id>&k=<k>
```

Returns a list of items similar to the given item.

**Parameters**:
- `item_id` (required): Item ID to find similar items for
- `k` (optional): Number of similar items to return (default: 10)

**Response**:
```json
[
  {
    "item_id": "item789",
    "score": 0.92,
    "title": "Similar Product",
    "category": "Category",
    "brand": "Brand",
    "price": 79.99,
    "img_url": "https://example.com/similar.jpg"
  },
  ...
]
```

## React Widget API

The React widget provides a responsive carousel for displaying recommendations:

```jsx
import { RecommendationCarousel } from 'recsys-lite-widget';

<RecommendationCarousel
  apiUrl="https://your-api-url"
  userId="user123"
  count={10}
  title="Recommended For You"
  className="custom-container"
  containerClassName="carousel-inner"
  cardClassName="product-card"
  onItemClick={(item) => console.log('Clicked item:', item)}
  fetchItemDetails={async (itemIds) => {
    // Optional: Fetch additional item details
    const response = await fetch(`/api/items?ids=${itemIds.join(',')}`);
    return await response.json();
  }}
/>
```

### Props

| Prop | Type | Description | Default |
|------|------|-------------|---------|
| `apiUrl` | `string` | Base URL of RecSys-Lite API | Required |
| `userId` | `string` | User ID to get recommendations for | Required |
| `count` | `number` | Number of recommendations to fetch | `10` |
| `title` | `string` | Title for the recommendations section | `"Recommended For You"` |
| `className` | `string` | CSS class for container | `""` |
| `containerClassName` | `string` | CSS class for carousel container | `""` |
| `cardClassName` | `string` | CSS class for product cards | `""` |
| `onItemClick` | `function` | Callback when a product is clicked | `undefined` |
| `fetchItemDetails` | `function` | Function to fetch additional item details | `undefined` |

## Docker Deployment

The system can be deployed using Docker:

```bash
# Start the complete system
docker compose -f docker/docker-compose.yml up -d

# Run tests
docker compose -f docker/docker-compose.test.yml up
```

## Data Contracts

### Events Data

Expected format for events data (Parquet):
- `user_id`: string - Unique user identifier
- `item_id`: string - Item identifier
- `ts`: int64 - Unix timestamp
- `qty`: int - Quantity purchased/viewed

### Items Data

Expected format for items data (CSV):
- `item_id`: string - Item identifier
- `category`: string - Item category
- `brand`: string - Item brand
- `price`: float - Item price
- `img_url`: string - URL to item image

## Development Guide

### Adding a New Model

1. Create a new model file in `src/recsys_lite/models/`
2. Implement the model class extending `BaseRecommender`
3. Update `src/recsys_lite/models/__init__.py` to export the new model
4. Add CLI command in `src/recsys_lite/cli.py`
5. Add tests in `tests/test_models.py`

### Running Tests

```bash
# Run all tests
pytest

# Run a specific test
pytest tests/test_models.py::test_als_model

# Run with coverage
pytest --cov=src tests/
```

## Operations Runbook

For detailed operational procedures, see [docs/runbook.md](docs/runbook.md).

## GDPR Compliance

RecSys-Lite is designed with GDPR compliance in mind:

- All data is stored locally, no external services are used
- User data can be exported and deleted via CLI commands
- All model updates are traceable
- Data residency is maintained within EU

## Requirements

- Python 3.11+
- DuckDB 0.8.1+
- CPU with 8+ cores
- 16GB+ RAM

## License

MIT

## Contributors

- RecSys-Lite Team