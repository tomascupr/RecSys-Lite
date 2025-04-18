"""BPR model implementation using implicit library."""

import os
import pickle
from typing import Any, Optional, Tuple, Union

import implicit
import numpy as np
import scipy.sparse as sp

from recsys_lite.models.base import BaseRecommender


class BPRModel(BaseRecommender):
    """Bayesian Personalized Ranking model for collaborative filtering."""

    def __init__(
        self,
        factors: int = 100,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize BPR model.

        Args:
            factors: Number of latent factors
            learning_rate: Learning rate for SGD
            regularization: Regularization factor
            iterations: Number of SGD iterations
            random_state: Random seed for reproducibility
        """
        self.model = implicit.bpr.BayesianPersonalizedRanking(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
            num_threads=0,  # Use all available cores
        )
        self.user_factors = None
        self.item_factors = None

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the BPR model.

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
        """Recommend items for a user.

        Args:
            user_id: User ID
            user_items: Sparse user-item interaction matrix
            n_items: Number of recommendations to return
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (item_ids, scores)
        """
        # Convert string user_id to int if needed
        if isinstance(user_id, str):
            user_id = int(user_id)

        # Get filter_already_liked_items parameter from kwargs or default to True
        filter_already_liked_items = kwargs.get("filter_already_liked_items", True)

        # BPR's recommend function expects user_items to have same number of rows as users
        # Extract just the relevant user's row
        user_row = user_items[user_id].copy()

        # Create a new matrix with just that one row
        single_user_matrix = sp.csr_matrix((1, user_items.shape[1]))
        single_user_matrix[0] = user_row

        # Get recommendations - implicit returns (item_ids, scores) tuple
        item_ids, scores = self.model.recommend(
            userid=0,  # Always use index 0 since we're only passing one user
            user_items=single_user_matrix,
            N=n_items,
            filter_already_liked_items=filter_already_liked_items,
        )

        return item_ids, scores

    def get_item_factors(self) -> np.ndarray:
        """Get item factors matrix.

        Returns:
            Item factors matrix
        """
        if self.item_factors is None:
            return np.array([])  # Return empty array if not initialized
        return self.item_factors

    def get_user_factors(self) -> np.ndarray:
        """Get user factors matrix.

        Returns:
            User factors matrix
        """
        if self.user_factors is None:
            return np.array([])  # Return empty array if not initialized
        return self.user_factors

    def save_model(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        os.makedirs(path, exist_ok=True)

        # Save model state
        model_state = {
            "factors": self.model.factors,
            "learning_rate": self.model.learning_rate,
            "regularization": self.model.regularization,
            "iterations": self.model.iterations,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
        }

        with open(os.path.join(path, "bpr_model.pkl"), "wb") as f:
            pickle.dump(model_state, f)

    def load_model(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        with open(os.path.join(path, "bpr_model.pkl"), "rb") as f:
            model_state = pickle.load(f)

        # Set model parameters
        self.model.factors = model_state["factors"]
        self.model.learning_rate = model_state["learning_rate"]
        self.model.regularization = model_state["regularization"]
        self.model.iterations = model_state["iterations"]

        # Set factors
        self.user_factors = model_state["user_factors"]
        self.item_factors = model_state["item_factors"]

        # Ensure model has these values too
        self.model.user_factors = self.user_factors
        self.model.item_factors = self.item_factors
