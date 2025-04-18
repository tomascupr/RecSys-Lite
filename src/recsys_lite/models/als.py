"""ALS model implementation using implicit library."""

from typing import Dict, Any, Tuple

import implicit
import numpy as np
import scipy.sparse as sp


class ALSModel:
    """Alternating Least Squares model for collaborative filtering."""
    
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
    
    def fit(self, user_item_matrix: sp.csr_matrix) -> None:
        """Fit the ALS model.
        
        Args:
            user_item_matrix: Sparse user-item interaction matrix
        """
        self.model.fit(user_item_matrix)
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors
    
    def partial_fit_users(self, user_item_matrix: sp.csr_matrix, user_ids: np.ndarray) -> None:
        """Update user factors for specified users.
        
        Args:
            user_item_matrix: Sparse user-item interaction matrix
            user_ids: IDs of users to update
        """
        self.model.partial_fit_users(user_item_matrix, user_ids)
        self.user_factors = self.model.user_factors
    
    def get_item_factors(self) -> np.ndarray:
        """Get item factors matrix.
        
        Returns:
            Item factors matrix
        """
        return self.item_factors