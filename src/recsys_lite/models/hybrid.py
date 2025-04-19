"""Hybrid recommendation model combining multiple base models.

This model provides a way to combine multiple recommendation models into a single
powerful recommender. It supports static or dynamic weighting of component models,
specialized handling for cold-start users, and efficient recommendation generation.

Key benefits:
- Combines content-based and collaborative filtering approaches
- Dynamic weighting based on user interaction patterns
- Optimal handling of cold-start users
- Efficient recommendation generation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from recsys_lite.models.base import BaseRecommender, ModelRegistry

logger = logging.getLogger("recsys-lite.models.hybrid")


class HybridModel(BaseRecommender):
    """Hybrid recommender combining multiple recommendation models."""

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
        """Initialize hybrid model.

        Args:
            models: List of recommender models
            weights: List of model weights (must sum to 1)
            dynamic_weighting: Whether to adjust weights based on user history
            cold_start_threshold: Number of interactions to consider cold-start
            content_models: List of model types considered content-based
            collaborative_models: List of model types considered collaborative
            cold_start_strategy: Strategy for cold-start users (content_boost|content_only|equal)
        """
        self.models = models or []
        self.weights = weights or [1.0 / len(self.models) if self.models else 1.0]
        self.dynamic_weighting = dynamic_weighting
        self.cold_start_threshold = cold_start_threshold
        self.content_models = content_models or ["text_embedding", "lightfm", "item2vec"]
        self.collaborative_models = collaborative_models or ["als", "bpr", "ease"]
        self.cold_start_strategy = cold_start_strategy

        # Validate weights
        if len(self.weights) != len(self.models):
            self.weights = [1.0 / len(self.models) for _ in self.models]

        # Normalize weights
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit all component models.

        Args:
            user_item_matrix: User-item interaction matrix
            **kwargs: Additional parameters passed to component models
        """
        for model in self.models:
            logger.info(f"Training component model: {model.model_type}")
            model.fit(user_item_matrix, **kwargs)

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate recommendations using all component models.

        Args:
            user_id: User ID
            user_items: User-item interaction matrix
            n_items: Number of recommendations
            **kwargs: Additional parameters
        """
        if not self.models:
            logger.warning("No component models available for recommendations")
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        # Determine weights (dynamic or static)
        weights = (
            self._get_dynamic_weights(user_id, user_items)
            if self.dynamic_weighting
            else self.weights
        )

        # Log weights for debugging
        if logger.isEnabledFor(logging.DEBUG):
            weight_info = ", ".join(
                [f"{m.model_type}:{w:.2f}" for m, w in zip(self.models, weights, strict=False)]
            )
            logger.debug(f"Model weights for user {user_id}: {weight_info}")

        # Get recommendations from each model
        all_item_scores: Dict[int, float] = {}

        # Request more items than needed to ensure diversity
        expanded_n = min(n_items * 3, 100)  # Request more items but cap at 100

        for model, weight in zip(self.models, weights, strict=False):
            # Skip models with zero weight
            if weight <= 0.001:  # Allow small threshold for floating point errors
                continue

            # Get recommendations from this model
            item_ids, scores = model.recommend(user_id, user_items, expanded_n, **kwargs)

            if len(item_ids) == 0:
                continue

            # Normalize scores to [0,1] range for fair combination
            if len(scores) > 0:
                min_score = scores.min()
                max_score = scores.max()
                if max_score > min_score:
                    normalized_scores = (scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.ones_like(scores)
            else:
                normalized_scores = scores

            # Add to aggregated scores with weight
            for idx, score in zip(item_ids, normalized_scores, strict=False):
                curr_score = all_item_scores.get(int(idx), 0.0)
                # Use max aggregation for scores from different models
                all_item_scores[int(idx)] = max(curr_score, float(score * weight))

        # Sort and select top items
        sorted_items = sorted(all_item_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:n_items]

        if not top_items:
            logger.warning(f"No recommendations generated for user {user_id}")
            return np.array([], dtype=np.int_), np.array([], dtype=np.float32)

        item_ids_list, scores_list = zip(*top_items, strict=False)

        # Convert to proper numpy arrays with the right types
        item_ids_arr = np.array(list(item_ids_list), dtype=np.int_)
        scores_arr = np.array(list(scores_list), dtype=np.float32)
        return item_ids_arr, scores_arr

    def _get_dynamic_weights(
        self, user_id: Union[int, str], user_items: sp.csr_matrix
    ) -> List[float]:
        """Dynamically adjust weights based on user interaction count.

        Args:
            user_id: User ID
            user_items: User-item interaction matrix

        Returns:
            Adjusted weights list
        """
        # Get user interaction count
        if isinstance(user_id, str) and user_id.isdigit():
            user_id = int(user_id)

        interaction_count = 0
        if isinstance(user_id, int) and user_id < user_items.shape[0]:
            user_row = user_items[user_id]
            interaction_count = user_row.nnz

        # Original weights
        adjusted_weights = self.weights.copy()

        # Adjust weights based on interaction count
        is_cold_start = interaction_count < self.cold_start_threshold

        if is_cold_start:
            # Find model indexes by type
            content_indices = []
            collab_indices = []

            for i, model in enumerate(self.models):
                if model.model_type in self.content_models:
                    content_indices.append(i)
                elif model.model_type in self.collaborative_models:
                    collab_indices.append(i)

            if self.cold_start_strategy == "content_boost" and content_indices and collab_indices:
                # Boost content models by transferring weight from collaborative models
                total_collab_weight = sum(adjusted_weights[i] for i in collab_indices)
                transfer = total_collab_weight * 0.7  # Transfer 70% of collaborative weight

                # Remove weight from collaborative models
                for i in collab_indices:
                    adjusted_weights[i] *= 0.3  # Keep 30% of original weight

                # Distribute transferred weight to content models proportionally
                total_content_weight = sum(adjusted_weights[i] for i in content_indices)
                if total_content_weight > 0:
                    for i in content_indices:
                        # Distribute proportionally to original weights
                        weight_ratio = adjusted_weights[i] / total_content_weight
                        adjusted_weights[i] += transfer * weight_ratio

            elif self.cold_start_strategy == "content_only" and content_indices:
                # Use only content-based models for cold start
                for i in range(len(self.models)):
                    adjusted_weights[i] = 1.0 if i in content_indices else 0.0

        # Normalize weights
        total = sum(adjusted_weights)
        if total > 0:
            adjusted_weights = [w / total for w in adjusted_weights]

        return adjusted_weights

    def get_item_vectors(self, item_ids: List[Union[str, int]]) -> np.ndarray:
        """Get item vectors - delegates to first model with vectors.

        Args:
            item_ids: List of item IDs

        Returns:
            Item vectors matrix
        """
        # Try to get vectors from content-based models first
        for model in self.models:
            if model.model_type in self.content_models and hasattr(model, "get_item_vectors"):
                vectors = model.get_item_vectors(item_ids)
                if isinstance(vectors, np.ndarray) and vectors.size > 0:
                    return vectors

        # Then try other models
        for model in self.models:
            if hasattr(model, "get_item_vectors"):
                vectors = model.get_item_vectors(item_ids)
                if isinstance(vectors, np.ndarray) and vectors.size > 0:
                    return vectors

        return np.array([])

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
                logger.info(f"Loaded component model: {model_type}")
            except Exception as e:
                logger.error(f"Error loading {model_type} model: {e}")
                continue

        # Validate weights
        if len(self.weights) != len(self.models):
            self.weights = [1.0 / len(self.models) for _ in self.models]

        # Normalize weights
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]


# Register the model
ModelRegistry.register("hybrid", HybridModel)
