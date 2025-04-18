"""Mock implementation of GRU4Rec for CI environments where PyTorch is unavailable."""

import os
import pickle
from typing import Any, List, Tuple, Union

import numpy as np
import scipy.sparse as sp

from recsys_lite.models.base import BaseRecommender


class GRU4Rec(BaseRecommender):
    """Mock GRU4Rec model for CI environment."""

    def __init__(
        self,
        n_items: int,
        hidden_size: int = 100,
        n_layers: int = 1,
        dropout: float = 0.1,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        n_epochs: int = 10,
        use_cuda: bool = False,
    ) -> None:
        """Initialize mock GRU4Rec model.

        Args:
            n_items: Number of items
            hidden_size: Size of hidden layers
            n_layers: Number of GRU layers
            dropout: Dropout probability
            batch_size: Training batch size
            learning_rate: Learning rate for Adam optimizer
            n_epochs: Number of training epochs
            use_cuda: Whether to use CUDA (GPU)
        """
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = "cpu"  # Always use CPU in mock

        # Placeholders for model state
        self.item_embeddings = None
        self._trained = False

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the model on user-item interaction data.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            **kwargs: Additional model-specific parameters
        """
        # Mock implementation - simply create random embeddings
        sessions = kwargs.get("sessions", [])
        if not sessions and user_item_matrix is not None:
            # If no sessions provided, try to create simple ones from matrix
            sessions = []
            for user_idx in range(user_item_matrix.shape[0]):
                items = user_item_matrix[user_idx].indices.tolist()
                if items:
                    sessions.append(items)

        # Create a mock item embeddings
        unique_items = set()
        for session in sessions:
            unique_items.update(session)

        self.n_items = max(self.n_items, len(unique_items) + 1)
        self.item_embeddings = np.random.randn(self.n_items, self.hidden_size).astype(
            np.float32
        )
        self._trained = True

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate recommendations for a user.

        Args:
            user_id: User ID
            user_items: Sparse user-item interaction matrix
            n_items: Number of recommendations to return
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (item_ids, scores)
        """
        # For GRU4Rec, we need a session, not just a user ID
        session = kwargs.get("session", [])
        return self.predict_next_items(session, n_items)

    def predict_next_items(
        self, session: List[int], k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next items for a session.

        Args:
            session: Current session sequence
            k: Number of items to recommend

        Returns:
            Tuple of (item_ids, scores)
        """
        if not self._trained:
            raise ValueError("Model has not been trained")

        # Generate random predictions for mock
        scores = np.random.randn(self.n_items)

        # Get top k items
        top_indices = np.argsort(-scores)[:k]
        top_scores = scores[top_indices]

        return top_indices, top_scores

    def save_model(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_state = {
            "n_items": self.n_items,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "item_embeddings": self.item_embeddings,
        }

        with open(path, "wb") as f:
            pickle.dump(model_state, f)

    def load_model(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        with open(path, "rb") as f:
            model_state = pickle.load(f)

        # Update model parameters
        self.n_items = model_state["n_items"]
        self.hidden_size = model_state["hidden_size"]
        self.n_layers = model_state["n_layers"]
        self.dropout = model_state["dropout"]
        self.batch_size = model_state["batch_size"]
        self.learning_rate = model_state["learning_rate"]
        self.n_epochs = model_state["n_epochs"]
        self.item_embeddings = model_state["item_embeddings"]
        self._trained = True
