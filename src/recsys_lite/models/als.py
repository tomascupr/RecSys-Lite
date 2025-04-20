"""ALS model implementation using implicit library."""

from typing import Any, Dict, Tuple, Union

import implicit
import numpy as np
import scipy.sparse as sp

from recsys_lite.models.base import BaseRecommender, FactorizationModelMixin


class ALSModel(BaseRecommender, FactorizationModelMixin):
    """Alternating Least Squares model for collaborative filtering."""

    model_type = "als"

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.01,
        alpha: float = 1.0,
        iterations: int = 15,
    ) -> None:
        """Initialize ALS model.

        Args:
            factors: Number of latent factors
            regularization: Regularization factor
            alpha: Confidence scaling parameter
            iterations: Number of ALS iterations
        """
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            calculate_training_loss=True,
            num_threads=0,  # Use all available cores
        )
        self.user_factors = None
        self.item_factors = None

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the ALS model.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            **kwargs: Additional model-specific parameters
        """
        self.model.fit(user_item_matrix)
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

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
        if isinstance(user_id, str):
            # Assume user_id is already an index if it's an integer
            # If it's a string, we need a mapping (not implemented here)
            user_id = int(user_id)

        # Get recommendations directly from implicit library
        recommendations = self.model.recommend(
            userid=user_id,
            user_items=user_items,
            N=n_items,
            filter_already_liked_items=True,
        )

        item_ids = np.array([item_id for item_id, _ in recommendations])
        scores = np.array([score for _, score in recommendations])

        return item_ids, scores

    def partial_fit_users(self, user_item_matrix: sp.csr_matrix, user_ids: np.ndarray) -> None:
        """Update user factors for specified users.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            user_ids: IDs of users to update
        """
        self.model.partial_fit_users(user_item_matrix, user_ids)
        self.user_factors = self.model.user_factors

    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization.

        Returns:
            Dictionary with model state
        """
        return {
            "factors": self.model.factors,
            "regularization": self.model.regularization,
            "alpha": self.model.alpha,
            "iterations": self.model.iterations,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
        }

    def _set_model_state(self, model_state: Dict[str, Any]) -> None:
        """Set model state from deserialized data.

        Args:
            model_state: Dictionary with model state
        """
        # Set model parameters
        self.model.factors = model_state["factors"]
        self.model.regularization = model_state["regularization"]
        self.model.alpha = model_state["alpha"]
        self.model.iterations = model_state["iterations"]

        # Set factors
        self.user_factors = model_state["user_factors"]
        self.item_factors = model_state["item_factors"]

        # Ensure model has these values too
        self.model.user_factors = self.user_factors
        self.model.item_factors = self.item_factors
