"""Mock implementation of GRU4Rec for CI environments where PyTorch is unavailable."""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from recsys_lite.models.base import BaseRecommender, ModelRegistry


class GRU4Rec(BaseRecommender):
    """Mock GRU4Rec model for CI environment."""

    model_type = "gru4rec"

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
        self.item_embeddings: Optional[NDArray[np.float32]] = None
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
        self.item_embeddings = np.random.randn(self.n_items, self.hidden_size).astype(np.float32)
        self._trained = True

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.int_], NDArray[np.float32]]:
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

    def predict_next_items(self, session: List[int], k: int = 10) -> Tuple[NDArray[np.int_], NDArray[np.float32]]:
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
        scores = np.random.randn(self.n_items).astype(np.float32)

        # Get top k items
        top_indices = np.argsort(-scores)[:k].astype(np.int_)
        top_scores = scores[top_indices].astype(np.float32)

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

    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "n_items": self.n_items,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "item_embeddings": self.item_embeddings,
            "_trained": self._trained,
        }

    def _set_model_state(self, model_state: Dict[str, Any]) -> None:
        """Set model state from deserialized data."""
        self.n_items = model_state.get("n_items", 0)
        self.hidden_size = model_state.get("hidden_size", 100)
        self.n_layers = model_state.get("n_layers", 1)
        self.dropout = model_state.get("dropout", 0.1)
        self.batch_size = model_state.get("batch_size", 32)
        self.learning_rate = model_state.get("learning_rate", 0.001)
        self.n_epochs = model_state.get("n_epochs", 10)
        self.item_embeddings = model_state.get("item_embeddings")
        self._trained = model_state.get("_trained", False)

    def get_item_vectors(self, item_ids: List[Union[str, int]]) -> NDArray[np.float32]:
        """Get item vectors for specified items."""
        if self.item_embeddings is None:
            return np.array([], dtype=np.float32)

        # For mock implementation, just return random vectors
        return np.random.randn(len(item_ids), self.hidden_size).astype(np.float32)


# Register the model
ModelRegistry.register("gru4rec", GRU4Rec)
