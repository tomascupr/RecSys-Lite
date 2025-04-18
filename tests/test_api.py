"""Tests for API service."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Skip tests in CI environment due to dependency issues
is_ci = os.environ.get("CI", "false").lower() == "true"
pytestmark = pytest.mark.skipif(
    is_ci, reason="Tests don't run in CI environment due to dependency issues"
)

from fastapi.testclient import TestClient
from recsys_lite.api.main import create_app
from recsys_lite.indexing import FaissIndexBuilder


@pytest.fixture
def mock_model_dir():
    """Create mock model artifacts for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create user and item mappings
        user_mapping = {f"U_{i}": i for i in range(10)}
        item_mapping = {f"I_{i}": i for i in range(20)}

        # Save mappings
        with open(temp_path / "user_mapping.json", "w") as f:
            json.dump(user_mapping, f)

        with open(temp_path / "item_mapping.json", "w") as f:
            json.dump(item_mapping, f)

        # Create mock Faiss index
        item_vectors = np.random.random((20, 10)).astype(np.float32)

        # Create Faiss index directory
        index_dir = temp_path / "faiss_index"
        index_dir.mkdir(exist_ok=True)

        # Create and save Faiss index
        index_builder = FaissIndexBuilder(
            vectors=item_vectors,
            ids=list(item_mapping.keys()),
            index_type="Flat",  # Use Flat for simplicity in tests
        )
        index_builder.save(index_dir)

        yield temp_path


@pytest.fixture
def mock_model(monkeypatch):
    """Mock model to use in tests."""
    # Create a mock model with necessary attributes
    mock = MagicMock()
    mock.user_factors = np.random.random((10, 10)).astype(np.float32)
    mock.item_factors = np.random.random((20, 10)).astype(np.float32)
    
    # Add recommend method to match BaseRecommender interface
    def mock_recommend(user_id, user_items, n_items):
        return [0, 1, 2, 3, 4], [0.9, 0.8, 0.7, 0.6, 0.5]
    mock.recommend = mock_recommend
    
    # Add get_user_representations and get_item_representations for LightFM
    def mock_get_user_representations():
        return (None, mock.user_factors)
    mock.get_user_representations = mock_get_user_representations
    
    def mock_get_item_representations():
        return (None, mock.item_factors)
    mock.get_item_representations = mock_get_item_representations
    
    # Mock the model loading function to return the mock model
    def mock_load(*args, **kwargs):
        return None
    mock.load_model = mock_load
    
    return mock

@pytest.fixture
def test_client(mock_model_dir, mock_model, monkeypatch):
    """Create an initialised ``TestClient`` for the FastAPI service.

    Starlette executes *startup* / *shutdown* lifespan events **only** when the
    client is used as a context manager.  Without those events the Faiss index
    stays un‑initialised and the endpoints would answer with *503 Service
    Unavailable*.  We therefore enter the context here and yield the ready‑to‑
    use instance to the calling test.
    """

    # ------------------------------------------------------------------
    #   Patch the heavy model imports inside the application start‑up so
    #   the tests remain lightweight – we simply return the prepared
    #   ``mock_model`` regardless of which concrete implementation the
    #   production code tries to load.
    # ------------------------------------------------------------------

    import sys
    from types import ModuleType

    fake_models_mod = ModuleType("recsys_lite.models")
    fake_models_mod.ALSModel = lambda *a, **k: mock_model
    fake_models_mod.BPRModel = lambda *a, **k: mock_model
    fake_models_mod.LightFMModel = lambda *a, **k: mock_model
    fake_models_mod.Item2VecModel = lambda *a, **k: mock_model
    fake_models_mod.GRU4Rec = lambda *a, **k: mock_model

    sys.modules["recsys_lite.models"] = fake_models_mod  # type: ignore[assignment]

    app = create_app(model_dir=mock_model_dir)

    # Enter the context manager so that *startup* is executed.
    with TestClient(app) as client:
        yield client


def test_health_endpoint(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_recommend_endpoint(test_client):
    """Test recommendation endpoint."""
    # Test with valid user ID
    response = test_client.get("/recommend?user_id=U_1&k=5")
    assert response.status_code == 200

    data = response.json()
    assert data["user_id"] == "U_1"
    assert "recommendations" in data
    assert len(data["recommendations"]) == 5

    # Check recommendation structure
    rec = data["recommendations"][0]
    assert "item_id" in rec
    assert "score" in rec

    # Test with invalid user ID
    response = test_client.get("/recommend?user_id=invalid_user&k=5")
    assert response.status_code == 404


def test_similar_items_endpoint(test_client):
    """Test similar items endpoint."""
    # Test with valid item ID
    response = test_client.get("/similar-items?item_id=I_1&k=5")
    assert response.status_code == 200

    recommendations = response.json()
    assert len(recommendations) <= 5  # May be less if some items are filtered

    # Check recommendation structure
    if recommendations:
        rec = recommendations[0]
        assert "item_id" in rec
        assert "score" in rec

    # Test with invalid item ID
    response = test_client.get("/similar-items?item_id=invalid_item&k=5")
    assert response.status_code == 404
