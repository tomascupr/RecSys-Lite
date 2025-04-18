"""BPR model implementation using implicit library."""

from typing import Dict, Any, Tuple, Optional

import implicit
import numpy as np
import scipy.sparse as sp


class BPRModel:
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
    
    def fit(self, user_item_matrix: sp.csr_matrix) -> None:
        """Fit the BPR model.
        
        Args:
            user_item_matrix: Sparse user-item interaction matrix
        """
        self.model.fit(user_item_matrix)
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors
    
    def recommend(
        self, 
        user_id: int, 
        user_items: sp.csr_matrix, 
        n_items: int = 10, 
        filter_already_liked_items: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recommend items for a user.
        
        Args:
            user_id: User ID
            user_items: Sparse user-item interaction matrix
            n_items: Number of recommendations to return
            filter_already_liked_items: Whether to filter out items the user has already interacted with
            
        Returns:
            Tuple of (item_ids, scores)
        """
        return self.model.recommend(
            userid=user_id,
            user_items=user_items,
            N=n_items,
            filter_already_liked_items=filter_already_liked_items,
        )
    
    def get_item_factors(self) -> np.ndarray:
        """Get item factors matrix.
        
        Returns:
            Item factors matrix
        """
        return self.item_factors
    
    def get_user_factors(self) -> np.ndarray:
        """Get user factors matrix.
        
        Returns:
            User factors matrix
        """
        return self.user_factors