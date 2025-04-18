"""LightFM model implementation for hybrid matrix factorization."""

from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import scipy.sparse as sp
from lightfm import LightFM


class LightFMModel:
    """Hybrid matrix factorization model using LightFM."""
    
    def __init__(
        self,
        no_components: int = 100,
        learning_rate: float = 0.05,
        loss: str = "warp",
        item_alpha: float = 0.0,
        user_alpha: float = 0.0,
        epochs: int = 50,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize LightFM model.
        
        Args:
            no_components: Number of latent factors
            learning_rate: Learning rate for SGD
            loss: Loss function to use ('warp', 'bpr', 'warp-kos', 'logistic')
            item_alpha: L2 penalty on item features
            user_alpha: L2 penalty on user features
            epochs: Number of training epochs
            random_state: Random seed for reproducibility
        """
        self.model = LightFM(
            no_components=no_components,
            learning_rate=learning_rate,
            loss=loss,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            random_state=random_state,
        )
        self.epochs = epochs
        self.user_biases = None
        self.item_biases = None
        self.user_embeddings = None
        self.item_embeddings = None
    
    def fit(
        self,
        interactions: sp.csr_matrix,
        user_features: Optional[sp.csr_matrix] = None,
        item_features: Optional[sp.csr_matrix] = None,
    ) -> None:
        """Fit the LightFM model.
        
        Args:
            interactions: Sparse user-item interaction matrix
            user_features: Sparse user features matrix
            item_features: Sparse item features matrix
        """
        self.model.fit(
            interactions=interactions,
            user_features=user_features,
            item_features=item_features,
            epochs=self.epochs,
            verbose=True,
        )
        
        # Store model parameters
        self.user_biases = self.model.user_biases
        self.item_biases = self.model.item_biases
        self.user_embeddings = self.model.user_embeddings
        self.item_embeddings = self.model.item_embeddings
    
    def predict(
        self, 
        user_ids: np.ndarray, 
        item_ids: np.ndarray,
        user_features: Optional[sp.csr_matrix] = None,
        item_features: Optional[sp.csr_matrix] = None,
    ) -> np.ndarray:
        """Predict scores for user-item pairs.
        
        Args:
            user_ids: User IDs
            item_ids: Item IDs
            user_features: Sparse user features matrix
            item_features: Sparse item features matrix
            
        Returns:
            Array of prediction scores
        """
        return self.model.predict(
            user_ids=user_ids,
            item_ids=item_ids,
            user_features=user_features,
            item_features=item_features,
        )
    
    def get_item_representations(
        self, item_features: Optional[sp.csr_matrix] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get item biases and embeddings.
        
        Args:
            item_features: Sparse item features matrix
            
        Returns:
            Tuple of (item_biases, item_embeddings)
        """
        if item_features is not None:
            return (
                self.model.item_biases,
                self.model.item_embeddings @ item_features.T
            )
        return self.model.item_biases, self.model.item_embeddings
    
    def get_user_representations(
        self, user_features: Optional[sp.csr_matrix] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get user biases and embeddings.
        
        Args:
            user_features: Sparse user features matrix
            
        Returns:
            Tuple of (user_biases, user_embeddings)
        """
        if user_features is not None:
            return (
                self.model.user_biases,
                self.model.user_embeddings @ user_features.T
            )
        return self.model.user_biases, self.model.user_embeddings
    
    def recommend_for_user(
        self,
        user_id: int,
        user_features: Optional[sp.csr_matrix] = None,
        item_features: Optional[sp.csr_matrix] = None,
        n_items: int = 10,
        filter_items: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recommend items for a user.
        
        Args:
            user_id: User ID
            user_features: Sparse user features matrix
            item_features: Sparse item features matrix
            n_items: Number of recommendations to return
            filter_items: List of item IDs to exclude from recommendations
            
        Returns:
            Tuple of (item_ids, scores)
        """
        # Get all item scores for the user
        n_items_total = self.item_embeddings.shape[0]
        scores = self.predict(
            user_ids=np.array([user_id] * n_items_total),
            item_ids=np.arange(n_items_total),
            user_features=user_features,
            item_features=item_features,
        )
        
        # Filter items if needed
        if filter_items:
            scores[filter_items] = -np.inf
            
        # Get top N items
        top_items = np.argsort(-scores)[:n_items]
        top_scores = scores[top_items]
        
        return top_items, top_scores