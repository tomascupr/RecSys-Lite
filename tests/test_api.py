"""Tests for API service."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip tests in CI environment due to dependency issues
is_ci = os.environ.get("CI", "false").lower() == "true"
pytestmark = pytest.mark.skipif(is_ci, reason="Tests don't run in CI environment due to dependency issues")

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
def test_client(mock_model_dir):
    """Create TestClient instance for testing the API."""
    app = create_app(model_dir=mock_model_dir)
    return TestClient(app)


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