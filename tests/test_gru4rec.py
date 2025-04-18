"""Tests for GRU4Rec model."""

import os
from pathlib import Path
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp
from unittest.mock import MagicMock, patch

from recsys_lite.models.gru4rec import GRU4Rec


# Skip in CI environment
is_ci = os.environ.get("CI", "false").lower() == "true"
pytestmark = pytest.mark.skipif(
    is_ci, reason="Tests with heavy dependencies don't run in CI environment"
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create a small user-item matrix
    n_users = 5
    n_items = 10
    
    # Create interaction matrix with some interactions
    rng = np.random.RandomState(42)
    interactions = sp.lil_matrix((n_users, n_items), dtype=np.float32)
    
    # Add some interactions
    for _ in range(15):
        user = rng.randint(0, n_users)
        item = rng.randint(0, n_items)
        interactions[user, item] = 1.0
    
    # Convert to CSR for efficient operations
    interactions = interactions.tocsr()
    
    # Create sessions for GRU4Rec
    sessions = []
    for user in range(n_users):
        items = interactions[user].nonzero()[1].tolist()
        if items:
            sessions.append(items)
    
    return interactions, sessions


@patch("recsys_lite.models.gru4rec.torch")
def test_gru4rec_initialization(mock_torch):
    """Test GRU4Rec model initialization."""
    # Initialize model
    model = GRU4Rec(
        n_items=100,
        hidden_size=50,
        n_layers=2,
        batch_size=32
    )
    
    # Check parameters
    assert model.n_items == 100
    assert model.batch_size == 32


@patch("recsys_lite.models.gru4rec.torch")
def test_gru4rec_fit(mock_torch, sample_data):
    """Test GRU4Rec model fit method."""
    interactions, sessions = sample_data
    
    # Mock the GRU4RecModel
    model = GRU4Rec(n_items=interactions.shape[1])
    model._fit_sessions = MagicMock(return_value={"loss": [0.5, 0.3]})
    
    # Call fit method
    model.fit(interactions, sessions=sessions)
    
    # Check that _fit_sessions was called
    model._fit_sessions.assert_called_once_with(sessions)


@patch("recsys_lite.models.gru4rec.torch")
def test_gru4rec_recommend(mock_torch):
    """Test GRU4Rec recommend method."""
    # Mock the model's predict_next_items method
    model = GRU4Rec(n_items=100)
    model.predict_next_items = MagicMock(
        return_value=(np.array([1, 3, 5]), np.array([0.9, 0.8, 0.7]))
    )
    
    # Create a user_items matrix
    user_items = sp.csr_matrix((1, 100))
    
    # Call recommend method
    items, scores = model.recommend(
        user_id=0,
        user_items=user_items,
        n_items=3,
        session=[2, 4]
    )
    
    # Check results
    assert np.array_equal(items, np.array([1, 3, 5]))
    assert np.array_equal(scores, np.array([0.9, 0.8, 0.7]))
    
    # Verify predict_next_items was called with the right arguments
    model.predict_next_items.assert_called_once_with([2, 4], 3)


@patch("recsys_lite.models.gru4rec.torch")
def test_gru4rec_save_load(mock_torch):
    """Test GRU4Rec model save and load methods."""
    # Mock torch.save and torch.load
    mock_torch.save = MagicMock()
    mock_torch.load = MagicMock(return_value={
        "model_state_dict": {"weights": "mock_weights"},
        "optimizer_state_dict": {"params": "mock_params"},
        "n_items": 200,
        "hidden_size": 128,
        "n_layers": 2,
    })
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile() as temp_file:
        # Initialize model
        model = GRU4Rec(n_items=100)
        model.model = MagicMock()
        model.model.state_dict = MagicMock(return_value={"weights": "mock_weights"})
        model.optimizer = MagicMock()
        model.optimizer.state_dict = MagicMock(
            return_value={"params": "mock_params"}
        )
        
        # Save model
        model.save_model(temp_file.name)
        
        # Check that torch.save was called
        mock_torch.save.assert_called_once()
        
        # Load model
        model.load_model(temp_file.name)
        
        # Check that torch.load was called
        mock_torch.load.assert_called_once()
        
        # Check that model parameters were updated
        assert model.n_items == 200
        assert model.model.load_state_dict.called
        assert model.optimizer.load_state_dict.called