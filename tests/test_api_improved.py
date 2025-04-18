"""Improved tests for RecSys-Lite API."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient

from recsys_lite.api.main import (
    Recommendation,
    RecommendationResponse,
    create_app,
    app as default_app
)


@pytest.fixture
def mock_model():
    """Create a mock model."""
    mock = MagicMock()
    # Set up recommend method to return some items and scores
    mock.recommend.return_value = (
        np.array([1, 2, 3]), 
        np.array([0.9, 0.8, 0.7])
    )
    return mock


@pytest.fixture
def mock_index():
    """Create a mock Faiss index."""
    mock = MagicMock()
    # Set up search method to return distances and indices
    mock.search.return_value = (
        np.array([[0.9, 0.8, 0.7]]), 
        np.array([[1, 2, 3]])
    )
    mock.d = 10  # Dimension of vectors
    return mock


@pytest.fixture
def test_app(mock_model, mock_index, tmp_path):
    """Create a test app with mocked components."""
    app = create_app(model_dir=tmp_path)
    
    # Create a test client
    client = TestClient(app)
    
    # Set up model and index
    app.state.model = mock_model
    app.state.faiss_index = mock_index
    app.state.model_type = "als"
    
    # Set up mappings
    app.state.user_mapping = {"user1": 0, "user2": 1}
    app.state.item_mapping = {"item1": 0, "item2": 1, "item3": 2, "item4": 3}
    app.state.reverse_user_mapping = {0: "user1", 1: "user2"}
    app.state.reverse_item_mapping = {0: "item1", 1: "item2", 2: "item3", 3: "item4"}
    
    # Set up item data
    app.state.item_data = {
        "item1": {"title": "Item 1", "category": "cat1"},
        "item2": {"title": "Item 2", "category": "cat2"},
        "item3": {"title": "Item 3", "category": "cat1"},
    }
    
    # Set up user-item matrix
    app.state.user_item_matrix = sp.csr_matrix((2, 4))
    
    return client


def test_health_endpoint(test_app):
    """Test the health endpoint."""
    response = test_app.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_metrics_endpoint(test_app):
    """Test the metrics endpoint."""
    response = test_app.get("/metrics")
    assert response.status_code == 200
    assert "uptime_seconds" in response.json()
    assert "request_count" in response.json()
    assert "model_type" in response.json()
    assert response.json()["model_type"] == "als"


def test_recommend_endpoint_with_faiss(test_app, mock_index):
    """Test the recommend endpoint with Faiss index."""
    # Make request
    response = test_app.get("/recommend?user_id=user1&k=3&use_faiss=true")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user1"
    assert len(data["recommendations"]) == 3
    
    # Check recommendations
    recs = data["recommendations"]
    assert recs[0]["item_id"] == "item2"  # From the reverse_item_mapping (index 1)
    assert recs[0]["score"] == 0.9
    assert recs[0]["title"] == "Item 2"
    
    # Verify mock was called properly
    mock_index.search.assert_called_once()


def test_recommend_endpoint_direct(test_app, mock_model):
    """Test the recommend endpoint with direct model recommendations."""
    # Make request
    response = test_app.get("/recommend?user_id=user1&k=3&use_faiss=false")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user1"
    assert len(data["recommendations"]) == 3
    
    # Verify mock was called properly
    mock_model.recommend.assert_called_once()


def test_recommend_endpoint_user_not_found(test_app):
    """Test the recommend endpoint with a user that doesn't exist."""
    response = test_app.get("/recommend?user_id=nonexistent&k=3")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_similar_items_endpoint(test_app, mock_index):
    """Test the similar-items endpoint."""
    # Make request
    response = test_app.get("/similar-items?item_id=item1&k=3")
    
    # Check response
    assert response.status_code == 200
    recs = response.json()
    assert len(recs) == 3
    
    # Check first recommendation
    assert recs[0]["item_id"] == "item2"  # From the reverse_item_mapping (index 1)
    assert recs[0]["score"] == 0.9
    
    # Verify mock was called properly
    mock_index.search.assert_called_once()


def test_similar_items_endpoint_item_not_found(test_app):
    """Test the similar-items endpoint with an item that doesn't exist."""
    response = test_app.get("/similar-items?item_id=nonexistent&k=3")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@patch("recsys_lite.api.main.FaissIndexBuilder")
def test_app_initialization(mock_faiss_builder, tmp_path):
    """Test app initialization loads model artifacts correctly."""
    # Create mock return from FaissIndexBuilder.load
    mock_index_builder = MagicMock()
    mock_index = MagicMock()
    mock_index_builder.index = mock_index
    mock_faiss_builder.load.return_value = mock_index_builder
    
    # Create test model directory with required artifacts
    model_dir = tmp_path / "als"
    model_dir.mkdir(parents=True)
    
    # Create user mapping
    with open(model_dir / "user_mapping.json", "w") as f:
        json.dump({"user1": 0, "user2": 1}, f)
    
    # Create item mapping
    with open(model_dir / "item_mapping.json", "w") as f:
        json.dump({"item1": 0, "item2": 1}, f)
    
    # Create model file (empty for test)
    (model_dir / "als_model.pkl").write_bytes(b"mock data")
    
    # Create a FastAPI test client
    with patch("recsys_lite.api.main.ALSModel") as mock_als:
        # Set up mock model
        mock_model = MagicMock()
        mock_als.return_value = mock_model
        
        # Create the app
        app = create_app(model_dir=model_dir)
        client = TestClient(app)
        
        # Trigger startup event manually
        app.router.startup()
        
        # Check that models were loaded
        mock_als.assert_called_once()
        mock_model.load_model.assert_called_once()
        mock_faiss_builder.load.assert_called_once_with(str(model_dir / "faiss_index"))
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}