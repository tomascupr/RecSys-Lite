"""Tests for LightFM model implementation."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.sparse as sp

from recsys_lite.models.lightfm_model import LightFMModel


# Skip in CI environment
is_ci = os.environ.get("CI", "false").lower() == "true"
pytestmark = pytest.mark.skipif(
    is_ci, reason="Tests with heavy dependencies don't run in CI environment"
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create a small user-item matrix
    n_users = 10
    n_items = 20
    
    # Create interaction matrix with some interactions
    rng = np.random.RandomState(42)
    interactions = sp.lil_matrix((n_users, n_items), dtype=np.float32)
    
    # Add some interactions (about 10% density)
    for _ in range(20):
        user = rng.randint(0, n_users)
        item = rng.randint(0, n_items)
        interactions[user, item] = 1.0
    
    # Convert to CSR for efficient operations
    interactions = interactions.tocsr()
    
    return interactions


@patch("recsys_lite.models.lightfm_model.LightFM")
def test_lightfm_model_initialization(mock_lightfm):
    """Test LightFMModel initialization."""
    # Initialize model
    model = LightFMModel(
        no_components=64,
        learning_rate=0.05,
        loss="warp",
        epochs=15
    )
    
    # Check parameters
    assert model.no_components == 64
    assert model.learning_rate == 0.05
    assert model.loss == "warp"
    assert model.epochs == 15
    
    # Check that LightFM was initialized with correct parameters
    mock_lightfm.assert_called_once_with(
        no_components=64,
        learning_rate=0.05,
        loss="warp"
    )


@patch("recsys_lite.models.lightfm_model.LightFM")
def test_lightfm_model_fit(mock_lightfm, sample_data):
    """Test LightFMModel fit method."""
    # Mock the LightFM model
    mock_model = MagicMock()
    mock_lightfm.return_value = mock_model
    
    # Initialize our wrapper
    model = LightFMModel()
    
    # Call fit method
    model.fit(sample_data)
    
    # Check that LightFM fit method was called
    mock_model.fit.assert_called_once()
    
    # Check that fit was called with the right arguments
    args, kwargs = mock_model.fit.call_args
    assert "item_features" in kwargs
    assert kwargs["item_features"] is None
    assert kwargs["epochs"] == 10  # Default value


@patch("recsys_lite.models.lightfm_model.LightFM")
def test_lightfm_model_predict(mock_lightfm, sample_data):
    """Test LightFMModel predict method."""
    # Mock the LightFM model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5, 0.8])
    mock_lightfm.return_value = mock_model
    
    # Initialize our wrapper
    model = LightFMModel()
    model.lightfm_model = mock_model
    
    # Set user and item embeddings
    model.user_embeddings = np.random.random((10, 32))
    model.item_embeddings = np.random.random((20, 32))
    
    # Call predict method
    user_ids = np.array([0, 1])
    item_ids = np.array([5, 8])
    scores = model.predict(user_ids, item_ids)
    
    # Check results
    assert np.array_equal(scores, np.array([0.5, 0.8]))
    
    # Check that LightFM predict method was called with right arguments
    mock_model.predict.assert_called_once()


@patch("recsys_lite.models.lightfm_model.LightFM")
def test_lightfm_model_recommend(mock_lightfm, sample_data):
    """Test LightFMModel recommend method."""
    # Mock the LightFM model
    mock_model = MagicMock()
    mock_lightfm.return_value = mock_model
    
    # Set up mock predict method to return scores for all items
    scores = np.zeros(sample_data.shape[1])
    scores[[2, 5, 8]] = [0.9, 0.8, 0.7]  # Top 3 items
    mock_model.predict.return_value = scores
    
    # Initialize our wrapper
    model = LightFMModel()
    model.lightfm_model = mock_model
    
    # Set user and item embeddings
    model.user_embeddings = np.random.random((10, 32))
    model.item_embeddings = np.random.random((20, 32))
    
    # Call recommend method
    user_id = 0
    user_items = sample_data
    n_items = 3
    items, rec_scores = model.recommend(user_id, user_items, n_items=n_items)
    
    # Check results
    assert np.array_equal(items, np.array([2, 5, 8]))
    assert np.array_equal(rec_scores, np.array([0.9, 0.8, 0.7]))


@patch("recsys_lite.models.lightfm_model.LightFM")
@patch("recsys_lite.models.lightfm_model.pickle")
def test_lightfm_model_save_load(mock_pickle, mock_lightfm):
    """Test LightFMModel save and load methods."""
    # Mock the pickle module
    mock_pickle.dump = MagicMock()
    mock_pickle.load = MagicMock(return_value={
        "model": "mock_model",
        "user_biases": np.array([0.1, 0.2]),
        "item_biases": np.array([0.3, 0.4, 0.5]),
        "user_embeddings": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "item_embeddings": np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]),
        "no_components": 64,
    })
    
    # Mock the LightFM model
    mock_model = MagicMock()
    mock_lightfm.return_value = mock_model
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile() as temp_file:
        # Initialize model
        model = LightFMModel()
        model.lightfm_model = mock_model
        
        # Save model
        model.save_model(temp_file.name)
        
        # Check that pickle.dump was called
        mock_pickle.dump.assert_called_once()
        
        # Load model
        model.load_model(temp_file.name)
        
        # Check that pickle.load was called
        mock_pickle.load.assert_called_once()
        
        # Check that model attributes were updated
        assert model.user_biases is not None
        assert model.item_biases is not None
        assert model.user_embeddings is not None
        assert model.item_embeddings is not None
        assert model.no_components == 64


@patch("recsys_lite.models.lightfm_model.LightFM")
def test_get_item_representations(mock_lightfm):
    """Test get_item_representations method."""
    # Mock the LightFM model
    mock_model = MagicMock()
    mock_lightfm.return_value = mock_model
    
    # Initialize our wrapper
    model = LightFMModel()
    
    # Set item embeddings and biases
    model.item_biases = np.array([0.1, 0.2, 0.3])
    model.item_embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    # Call method
    biases, vectors = model.get_item_representations()
    
    # Check results
    assert np.array_equal(biases, np.array([0.1, 0.2, 0.3]))
    assert np.array_equal(vectors, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))