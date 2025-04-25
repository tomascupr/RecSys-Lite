# RecSys-Lite

A lightweight recommendation system designed for small e-commerce shops running on CPU-only environments. RecSys-Lite provides multiple recommendation algorithms, automatic hyperparameter optimization, fast recommendation serving via Faiss, and a React widget for displaying recommendations.

![RecSys-Lite](https://img.shields.io/badge/RecSys--Lite-v0.3.0-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green)
![React](https://img.shields.io/badge/React-18-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

## Features

- **Multiple Recommendation Algorithms**:
  - ALS and BPR via the `implicit` library
  - Item2Vec embeddings via `gensim`
  - Hybrid matrix factorization via `LightFM`
  - GRU4Rec session-based model via PyTorch
  - EASE‑R linear item‑item model (state‑of‑the‑art accuracy, CPU friendly)
  - Text embedding model via `sentence-transformers` (all-MiniLM-L6-v2)
  - Hybrid model for combining collaborative filtering with content-based approaches

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

Additional documentation:
- [Logging System](docs/logging.md)
- [Error Handling](docs/error_handling.md)
- [Vector Service](docs/vector_service.md)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tomascupr/recsys-lite.git
cd recsys-lite

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies using Poetry (recommended)
poetry install

# Or install with LLM support
poetry install -E llm

# Or install using pip
pip install -e .

# Or with LLM support
pip install -e ".[llm]"

# Or use Docker
docker compose -f docker/docker-compose.yml up -d
```

### Data Ingestion

```bash
# Ingest data into DuckDB
recsys-lite ingest --events path/to/events.parquet --items path/to/items.csv --db recsys.db

# File-based streaming: continuously append new *.parquet files that appear in a directory
recsys-lite stream-ingest data/incremental --db recsys.db --poll-interval 5

# Message queue-based streaming (requires additional dependencies)
# Install message queue dependencies: pip install recsys-lite[mq]

# RabbitMQ streaming
recsys-lite queue-ingest rabbitmq --db recsys.db --rabbitmq-host localhost --rabbitmq-queue events

# Kafka streaming
recsys-lite queue-ingest kafka --db recsys.db --kafka-servers localhost:9092 --kafka-topic events
```

### Training Models

```bash
# Train ALS model
recsys-lite train als --db recsys.db --output model_artifacts/als

# Train item2vec model
recsys-lite train item2vec --db recsys.db --output model_artifacts/item2vec

# Train EASE-R model
recsys-lite train ease --db recsys.db --output model_artifacts/ease

# Train text embedding model (requires sentence-transformers)
recsys-lite train text_embedding --db recsys.db --output model_artifacts/text_embedding

# Create a hybrid model (combining ALS and text embedding)
recsys-lite train_hybrid model_artifacts/als model_artifacts/text_embedding --output model_artifacts/hybrid

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

## Development

RecSys-Lite is actively developed with focus on lightweight recommendation algorithms, quality, and maintainability.

```bash
# Install development dependencies
poetry install

# Run linting
poetry run ruff .

# Run tests
poetry run pytest
```

## CLI Commands Reference

### Data Management

| Command | Description | Options |
|---------|-------------|---------|
| `ingest` | Ingest data into DuckDB | `--events <parquet>`, `--items <csv>`, `--db <path>` |
| `stream-ingest` | Watch a directory and append new parquet files to `events` table | `--poll-interval <sec>`, `--db <path>` |
| `queue-ingest` | Stream events from a message queue to `events` table | `rabbitmq|kafka`, `--db <path>`, `--batch-size <num>` |
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
| `train ease` | Train EASE‑R linear item‑based model | `--db <path>`, `--output <dir>` |
| `train text_embedding` | Train text embedding model | `--db <path>`, `--output <dir>`, `--model-name <name>`, `--fields <fields>` |
| `train_hybrid` | Create hybrid model | `models_dir [models_dir...]`, `--output <dir>`, `--weights <w1,w2...>`, `--dynamic`, `--cold-start-threshold <n>` |

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

### Cache Management Endpoints

The API includes endpoints for managing the recommendation caches:

```
POST /cache/clear
```
Clears all caches (recommendation and similar-items caches)

```
POST /cache/config?enabled=<true|false>&default_ttl=<seconds>
```
Configure cache settings globally

**Parameters**:
- `enabled` (required): Whether to enable caching (true/false)
- `default_ttl` (optional): Default TTL in seconds for all caches (0-3600, default: 300)

**Response**:
```json
{
  "status": "success",
  "cache_enabled": true,
  "default_ttl": 300,
  "recommendation_cache": {
    "size": 10,
    "hits": 45,
    "misses": 12,
    "hit_rate": 0.789
  },
  "similar_items_cache": {
    "size": 5,
    "hits": 23,
    "misses": 8,
    "hit_rate": 0.742
  }
}
```

### Recommendation Endpoint

```
GET /recommend?user_id=<user_id>&k=<k>&use_faiss=<true|false>&page=<page>&page_size=<page_size>&categories=<cat1>&categories=<cat2>&brands=<brand1>&min_price=<min>&max_price=<max>&exclude_items=<item1>&include_items=<item2>&rerank=<true|false>&rerank_strategy=<strategy>&diversity_weight=<0-1>&use_cache=<true|false>&cache_ttl=<seconds>
```

Returns a list of recommended items for a user with pagination, filtering, personalized reranking, and caching support.

**Parameters**:
- `user_id` (required): User ID to get recommendations for
- `k` (optional): Number of recommendations to return (default: 10)
- `use_faiss` (optional): Whether to use Faiss index for similarity search (default: true)

**Pagination Parameters**:
- `page` (optional): Page number, 1-based (default: 1)
- `page_size` (optional): Number of items per page, max 100 (default: 10)

**Filtering Parameters**:
- `categories` (optional): Filter by categories (can specify multiple)
- `brands` (optional): Filter by brands (can specify multiple)
- `min_price` (optional): Minimum price filter
- `max_price` (optional): Maximum price filter
- `exclude_items` (optional): Item IDs to exclude (can specify multiple)
- `include_items` (optional): Limit to these item IDs (can specify multiple)

**Reranking Parameters**:
- `rerank` (optional): Whether to apply reranking strategy (default: false)
- `rerank_strategy` (optional): Reranking strategy to use: diversity, novelty, popularity, or hybrid (default: hybrid)
- `diversity_weight` (optional): Weight for diversity component, 0-1 (default: 0.2)
- `novelty_weight` (optional): Weight for novelty component, 0-1 (default: 0.2)
- `popularity_weight` (optional): Weight for popularity component, 0-1 (default: 0.2)
- `relevance_weight` (optional): Weight for baseline relevance, 0-1 (default: 0.4)

**Cache Parameters**:
- `use_cache` (optional): Whether to use cache if available (default: true)
- `cache_ttl` (optional): Cache time-to-live in seconds, 0-3600 (default: 300)
- `invalidate_cache` (optional): Force refresh of cache for this request (default: false)

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
      "img_url": "https://example.com/image.jpg",
      "explanation": "Recommended based on your interests"
    },
    {
      "item_id": "item789",
      "score": 0.87,
      "title": "Another Product",
      "category": "Clothing",
      "brand": "FashionCo", 
      "price": 49.99,
      "img_url": "https://example.com/image2.jpg",
      "explanation": "Adding variety with this Clothing item"
    },
    ...
  ],
  "pagination": {
    "total": 120,
    "page": 1,
    "page_size": 10,
    "total_pages": 12,
    "has_next": true,
    "has_prev": false
  },
  "filter_info": {
    "original_count": 50,
    "filtered_count": 25,
    "filters_applied": {
      "categories": ["Electronics", "Tech"],
      "min_price": 10.0,
      "max_price": 100.0
    },
    "cache": {
      "hit": true,
      "ttl": 300,
      "invalidated": false
    },
    "reranking": {
      "strategy": "hybrid",
      "applied": true,
      "weights": {
        "diversity": 0.2,
        "novelty": 0.2,
        "popularity": 0.2,
        "relevance": 0.4
      },
      "category_distribution": {
        "Electronics": 4,
        "Clothing": 2,
        "Home": 3,
        "Books": 1
      }
    }
  }
}
```

**Error Responses**:
- `404 Not Found`: User ID not found
- `400 Bad Request`: Invalid parameters
- `500 Internal Server Error`: Server error

### Similar Items Endpoint

```
GET /similar-items?item_id=<item_id>&k=<k>&page=<page>&page_size=<page_size>&categories=<cat1>&brands=<brand1>&min_price=<min>&max_price=<max>&exclude_items=<item1>&rerank=<true|false>&rerank_strategy=<strategy>&diversity_weight=<0-1>&use_cache=<true|false>&cache_ttl=<seconds>
```

Returns a list of items similar to the given item with pagination, filtering, personalized reranking and caching support.

**Parameters**:
- `item_id` (required): Item ID to find similar items for
- `k` (optional): Number of similar items to return (default: 10)

**Pagination Parameters**:
- `page` (optional): Page number, 1-based (default: 1)
- `page_size` (optional): Number of items per page, max 100 (default: 10)

**Filtering Parameters**:
- `categories` (optional): Filter by categories (can specify multiple)
- `brands` (optional): Filter by brands (can specify multiple)
- `min_price` (optional): Minimum price filter
- `max_price` (optional): Maximum price filter
- `exclude_items` (optional): Item IDs to exclude (can specify multiple)

**Reranking Parameters**:
- `rerank` (optional): Whether to apply reranking strategy (default: false)
- `rerank_strategy` (optional): Reranking strategy to use: diversity, novelty, popularity, or hybrid (default: diversity)
- `diversity_weight` (optional): Weight for diversity component, 0-1 (default: 0.3)
- `novelty_weight` (optional): Weight for novelty component, 0-1 (default: 0.1)
- `popularity_weight` (optional): Weight for popularity component, 0-1 (default: 0.2)
- `relevance_weight` (optional): Weight for baseline relevance, 0-1 (default: 0.4)

**Cache Parameters**:
- `use_cache` (optional): Whether to use cache if available (default: true)
- `cache_ttl` (optional): Cache time-to-live in seconds, 0-3600 (default: 300)
- `invalidate_cache` (optional): Force refresh of cache for this request (default: false)

**Response**:
```json
{
  "user_id": "item123",
  "recommendations": [
    {
      "item_id": "item789",
      "score": 0.92,
      "title": "Similar Product",
      "category": "Category",
      "brand": "Brand",
      "price": 79.99,
      "img_url": "https://example.com/similar.jpg",
      "explanation": "Adding variety with this Category item"
    },
    ...
  ],
  "pagination": {
    "total": 42,
    "page": 1,
    "page_size": 10,
    "total_pages": 5,
    "has_next": true,
    "has_prev": false
  },
  "filter_info": {
    "original_count": 30,
    "filtered_count": 15,
    "filters_applied": {
      "brands": ["BrandName"],
      "excluded_items": 2
    },
    "cache": {
      "hit": true,
      "ttl": 300,
      "invalidated": false
    },
    "reranking": {
      "strategy": "diversity",
      "applied": true,
      "weights": {
        "diversity": 0.3,
        "novelty": 0.1,
        "popularity": 0.2,
        "relevance": 0.4
      },
      "category_distribution": {
        "Category": 3,
        "OtherCategory": 2,
        "ThirdCategory": 5
      }
    }
  }
}
```

**Error Responses**:
- `404 Not Found`: Item ID not found
- `400 Bad Request`: Invalid parameters
- `500 Internal Server Error`: Server error

### Health Check Endpoint

```
GET /health
```

Returns the health status of the API.

**Response**:
```json
{
  "status": "ok",
  "version": "0.2.0",
  "model_type": "als",
  "uptime_seconds": 3600
}
```

### Metrics Endpoint

```
GET /metrics
```

Returns service metrics for monitoring.

**Response**:
```json
{
  "uptime_seconds": 3600,
  "request_count": 1500,
  "recommendation_count": 15000,
  "error_count": 5,
  "recommendations_per_second": 4.16,
  "cache_hit_ratio": 0.85
}
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

### Widget Customization

You can customize the appearance of the widget using CSS classes:

```css
/* Example custom styling */
.custom-container {
  max-width: 1200px;
  margin: 0 auto;
}

.carousel-inner {
  padding: 10px 0;
}

.product-card {
  border-radius: 8px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;
}

.product-card:hover {
  transform: translateY(-5px);
}
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
# Create directories for persistence
mkdir -p data/incremental model_artifacts

# Start the complete system
docker compose -f docker/docker-compose.yml up -d

# Run tests
docker compose -f docker/docker-compose.test.yml up
```

### Volume Mapping

For data persistence with Docker, map these volumes:

```yaml
volumes:
  - ./data:/data                         # Database and event data
  - ./model_artifacts:/app/model_artifacts  # Trained models
  - ./logs:/var/log                       # Log files
```

## Data Contracts

### Events Data

Expected format for events data (Parquet):
- `user_id`: string - Unique user identifier
- `item_id`: string - Item identifier
- `ts`: int64 - Unix timestamp
- `qty`: int - Quantity purchased/viewed

Sample data structure:
```
┌────────────┬────────────┬────────────┬──────┐
│  user_id   │  item_id   │     ts     │ qty  │
├────────────┼────────────┼────────────┼──────┤
│ user_12345 │ item_67890 │ 1626533119 │ 1    │
│ user_12345 │ item_24680 │ 1626533135 │ 2    │
│ user_67890 │ item_13579 │ 1626533257 │ 1    │
└────────────┴────────────┴────────────┴──────┘
```

### Items Data

Expected format for items data (CSV):
- `item_id`: string - Item identifier
- `category`: string - Item category
- `brand`: string - Item brand
- `price`: float - Item price
- `img_url`: string - URL to item image

Sample data structure:
```
┌────────────┬──────────────┬────────────┬───────┬───────────────────────────────┐
│  item_id   │   category   │   brand    │ price │           img_url             │
├────────────┼──────────────┼────────────┼───────┼───────────────────────────────┤
│ item_67890 │ Electronics  │ TechBrand  │ 99.99 │ https://example.com/img1.jpg  │
│ item_24680 │ Clothing     │ FashionCo  │ 49.99 │ https://example.com/img2.jpg  │
│ item_13579 │ Home         │ HomeMakers │ 29.99 │ https://example.com/img3.jpg  │
└────────────┴──────────────┴────────────┴───────┴───────────────────────────────┘
```

## Development Guide

### Adding a New Model

1. Create a new model file in `src/recsys_lite/models/`
2. Implement the model class extending `BaseRecommender`
3. Update `src/recsys_lite/models/__init__.py` to export the new model
4. Add a mock implementation for CI environment if needed
5. Add the model type to the `ModelType` enum in `src/recsys_lite/cli.py`
6. Add CLI command in `src/recsys_lite/cli.py`
7. Add tests in `tests/test_models.py`

Example model implementation:
```python
from recsys_lite.models.base import BaseRecommender

class MyNewModel(BaseRecommender):
    def __init__(self, params=None):
        super().__init__(params)
        self.params = params or {}
        
    def fit(self, user_items):
        # Implementation logic
        pass
        
    def recommend(self, user_id, n=10):
        # Implementation logic
        pass
        
    def similar_items(self, item_id, n=10):
        # Implementation logic
        pass
        
    def save(self, path):
        # Implementation logic
        pass
        
    @classmethod
    def load(cls, path):
        # Implementation logic
        pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run a specific test
pytest tests/test_models.py::test_als_model

# Run with coverage
pytest --cov=src tests/

# Run linting
ruff .

# Run type checking
mypy .
```

## GDPR Compliance

RecSys-Lite is designed with GDPR compliance in mind:

### User Data Export

The `gdpr export-user` command exports all data associated with a user, including:
- Raw event data (interactions with items)
- Timestamps of interactions
- Item metadata for interacted items

The export is provided in JSON format and includes all information stored about the user.

### User Data Deletion

The `gdpr delete-user` command removes all user data from the system:
1. Deletes raw events from the database
2. Marks the user for removal from models
3. Removes user vectors during the next update cycle

After deletion, the user will no longer receive personalized recommendations.

### Audit Trail

All GDPR-related operations are logged with timestamps, including:
- Data export requests
- Data deletion requests
- Model updates after deletion
- API access patterns

## Environment Variables

RecSys-Lite can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_DIR` | Directory for model artifacts | `model_artifacts/als` |
| `DB_PATH` | Path to DuckDB database | `data/recsys.db` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_RECOMMENDATIONS` | Maximum recommendations to return | `100` |
| `CACHE_EXPIRY` | Cache expiry time in seconds | `300` |
| `FAISS_NPROBE` | Number of clusters to probe in Faiss | `10` |
| `BATCH_SIZE` | Batch size for processing | `1000` |
| `THREADS` | Number of threads to use | `8` |
| `FAISS_USE_THREADS` | Whether to use threads in Faiss | `1` |

## Troubleshooting

### Common Issues

**API returns "Model not found" error**:
- Check that the model directory exists and contains all required files
- Verify the model was trained successfully
- Try retraining the model

**Poor recommendation quality**:
- Increase the number of factors in the model
- Run hyperparameter optimization
- Use a hybrid model like LightFM with item features
- Check data quality and increase training data volume

**Update worker not updating recommendations**:
- Check worker logs for errors
- Verify incremental data is in the correct format
- Check database connectivity

**High memory usage**:
- Reduce the number of factors in models
- Decrease FAISS_NPROBE value
- Reduce BATCH_SIZE for updates
- Consider using a smaller model like ALS instead of LightFM

## Requirements

- Python 3.11+
- DuckDB 0.8.1+
- CPU with 8+ cores
- 16GB+ RAM

## License

Apache License 2.0

## Contributing

We welcome contributions to RecSys-Lite! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

For more details, see our [Contributing Guide](CONTRIBUTING.md).

## Contributors

RecSys-Lite is maintained by the RecSys-Lite Team. Special thanks to all contributors!