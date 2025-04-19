"""Mock implementation of hybrid model for testing and CI environment."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from recsys_lite.models.base import BaseRecommender, ModelRegistry


class HybridModel(BaseRecommender):
    """Mock hybrid model for testing purposes."""

    model_type = "hybrid"

    def __init__(
        self,
        models: Optional[List[BaseRecommender]] = None,
        weights: Optional[List[float]] = None,
        dynamic_weighting: bool = True,
        cold_start_threshold: int = 5,
        content_models: Optional[List[str]] = None,
        collaborative_models: Optional[List[str]] = None,
        cold_start_strategy: str = "content_boost",
    ) -> None:
        """Initialize mock hybrid model."""
        self.models = models or []
        self.weights = weights or [1.0 / len(self.models) if self.models else 1.0]
        self.dynamic_weighting = dynamic_weighting
        self.cold_start_threshold = cold_start_threshold
        self.content_models = content_models or ["text_embedding", "lightfm", "item2vec"]
        self.collaborative_models = collaborative_models or ["als", "bpr", "ease"]
        self.cold_start_strategy = cold_start_strategy

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Mock fit implementation."""
        # In mock, we fit each component model
        for model in self.models:
            model.fit(user_item_matrix, **kwargs)

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.int_], NDArray[np.float32]]:
        """Mock recommend implementation.

        In this mock version, we just delegate to the first model or return empty arrays.
        """
        if not self.models:
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        # Just use the first model for simplicity in mock
        return self.models[0].recommend(user_id, user_items, n_items, **kwargs)

    def get_item_vectors(self, item_ids: List[Union[str, int]]) -> NDArray[np.float32]:
        """Get item vectors from the first model with vectors."""
        for model in self.models:
            if hasattr(model, "get_item_vectors"):
                vectors = model.get_item_vectors(item_ids)
                if isinstance(vectors, np.ndarray) and vectors.size > 0:
                    return vectors

        return np.array([], dtype=np.float32)

    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        model_states = []
        for model in self.models:
            model_type = model.model_type
            model_state = model._get_model_state()
            model_states.append((model_type, model_state))

        return {
            "model_states": model_states,
            "weights": self.weights,
            "dynamic_weighting": self.dynamic_weighting,
            "cold_start_threshold": self.cold_start_threshold,
            "content_models": self.content_models,
            "collaborative_models": self.collaborative_models,
            "cold_start_strategy": self.cold_start_strategy,
        }

    def _set_model_state(self, model_state: Dict[str, Any]) -> None:
        """Set model state from deserialized data."""
        self.weights = model_state.get("weights", [])
        self.dynamic_weighting = model_state.get("dynamic_weighting", True)
        self.cold_start_threshold = model_state.get("cold_start_threshold", 5)
        self.content_models = model_state.get(
            "content_models", ["text_embedding", "lightfm", "item2vec"]
        )
        self.collaborative_models = model_state.get("collaborative_models", ["als", "bpr", "ease"])
        self.cold_start_strategy = model_state.get("cold_start_strategy", "content_boost")

        # Initialize component models
        self.models = []
        for model_type, state in model_state.get("model_states", []):
            try:
                model = ModelRegistry.create_model(model_type)
                model._set_model_state(state)
                self.models.append(model)
            except Exception:
                continue
