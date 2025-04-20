"""Mock implementation of text embedding model for testing and CI environment."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from recsys_lite.models.base import BaseRecommender


class TextEmbeddingModel(BaseRecommender):
    """Mock text embedding model for testing purposes."""

    model_type = "text_embedding"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        item_text_fields: Optional[List[str]] = None,
        field_weights: Optional[Dict[str, float]] = None,
        normalize_vectors: bool = True,
        cache_embeddings: bool = True,
        batch_size: int = 32,
        max_length: int = 512,
        embedding_dim: int = 384,
    ) -> None:
        """Initialize mock text embedding model."""
        self.model_name = model_name
        self.item_text_fields = item_text_fields or ["title", "category", "brand", "description"]
        self.field_weights = field_weights or {
            "title": 2.0,
            "category": 1.0,
            "brand": 1.0,
            "description": 3.0,
        }
        self.normalize_vectors = normalize_vectors
        self.cache_embeddings = cache_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        # Mock embeddings with fixed dimension
        self.item_embeddings: Optional[NDArray[np.float32]] = None
        self.item_ids: Optional[List[str]] = None
        self.id_to_idx: Dict[str, int] = {}

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Generate mock item embeddings."""
        # Extract required fields for mock
        item_data = kwargs.get("item_data", {})

        # Create random embeddings for items
        n_items = len(item_data)
        if n_items > 0:
            # Create deterministic random embeddings
            np.random.seed(42)
            self.item_embeddings = np.random.randn(n_items, self.embedding_dim).astype(np.float32)

            # Normalize if requested
            if self.normalize_vectors:
                norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
                self.item_embeddings = self.item_embeddings / np.maximum(norms, 1e-12)

            # Store item IDs
            self.item_ids = list(str(key) for key in item_data.keys())

            # Create mapping
            self.id_to_idx = {id_: i for i, id_ in enumerate(self.item_ids or [])}

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.int_], NDArray[np.float32]]:
        """Generate mock recommendations."""
        if self.item_embeddings is None or self.item_ids is None:
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        item_mapping = kwargs.get("item_mapping", {})
        reverse_item_mapping = kwargs.get("reverse_item_mapping", {})

        # Get a deterministic sequence of items based on user_id
        if isinstance(user_id, str) and user_id.isdigit():
            user_id = int(user_id)

        # Create deterministic sequence
        np.random.seed(user_id if isinstance(user_id, int) else 0)

        # Get interacted items to exclude
        interacted_item_ids = []
        if isinstance(user_id, int) and user_id < user_items.shape[0]:
            interacted_indices = user_items[user_id].indices
            for idx in interacted_indices:
                item_id = reverse_item_mapping.get(int(idx))
                if item_id and item_id in self.id_to_idx:
                    interacted_item_ids.append(item_id)

        # Generate recommendations
        all_item_indices = list(range(len(self.item_ids)))
        np.random.shuffle(all_item_indices)

        # Filter out interacted items
        filtered_indices = [i for i in all_item_indices if self.item_ids[i] not in interacted_item_ids]

        # Get top N items
        top_indices = filtered_indices[:n_items]

        # Convert to global item indices
        top_item_indices = []
        for idx in top_indices:
            item_id = self.item_ids[idx]
            if item_id in item_mapping:
                top_item_indices.append(item_mapping[item_id])

        # Generate mock scores
        scores = np.linspace(0.9, 0.5, len(top_item_indices)).astype(np.float32)

        return np.array(top_item_indices, dtype=np.int_), scores

    def get_item_vectors(self, item_ids: List[Union[str, int]]) -> NDArray[np.float32]:
        """Get mock item vectors."""
        if self.item_embeddings is None or self.item_ids is None:
            return np.array([], dtype=np.float32)

        indices: List[int] = []
        for item_id in item_ids:
            item_id_str = str(item_id)
            if item_id_str in self.id_to_idx:
                indices.append(self.id_to_idx[item_id_str])

        if indices and self.item_embeddings is not None:
            return self.item_embeddings[indices]
        return np.array([], dtype=np.float32)

    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "model_name": self.model_name,
            "item_text_fields": self.item_text_fields,
            "field_weights": self.field_weights,
            "normalize_vectors": self.normalize_vectors,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "item_embeddings": self.item_embeddings,
            "item_ids": self.item_ids,
        }

    def _set_model_state(self, model_state: Dict[str, Any]) -> None:
        """Set model state from deserialized data."""
        self.model_name = model_state["model_name"]
        self.item_text_fields = model_state["item_text_fields"]
        self.field_weights = model_state.get("field_weights", {})
        self.normalize_vectors = model_state["normalize_vectors"]
        self.batch_size = model_state.get("batch_size", 32)
        self.max_length = model_state.get("max_length", 512)
        self.embedding_dim = model_state.get("embedding_dim", 384)
        self.item_embeddings = model_state.get("item_embeddings")
        self.item_ids = model_state.get("item_ids")

        # Recreate id to index mapping
        if self.item_ids:
            self.id_to_idx = {id_: i for i, id_ in enumerate(self.item_ids)}
