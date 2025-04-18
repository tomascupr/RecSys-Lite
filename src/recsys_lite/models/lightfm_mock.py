"""Mock version of LightFM model to use in CI where LightFM fails to build."""

import os
import pickle
from typing import Dict, Any, Tuple, Optional, List, Union, cast

import numpy as np
import scipy.sparse as sp

from recsys_lite.models.base import BaseRecommender


class LightFMModel(BaseRecommender):
    """Mock hybrid matrix factorization model for CI environment."""
    
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
        """Initialize mock LightFM model.
        
        Args:
            no_components: Number of latent factors
            learning_rate: Learning rate for SGD
            loss: Loss function to use ('warp', 'bpr', 'warp-kos', 'logistic')
            item_alpha: L2 penalty on item features
            user_alpha: L2 penalty on user features
            epochs: Number of training epochs
            random_state: Random seed for reproducibility
        """
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.loss = loss
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.epochs = epochs
        self.random_state = random_state
        self.user_biases = None
        self.item_biases = None
        self.user_embeddings = None
        self.item_embeddings = None
        self.user_features = None
        self.item_features = None
    
    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the mock model on user-item interaction data.
        
        Args:
            user_item_matrix: Sparse user-item interaction matrix
            **kwargs: Additional model-specific parameters
        """
        # Mock implementation - just create random embeddings
        n_users, n_items = user_item_matrix.shape
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.user_embeddings = np.random.rand(n_users, self.no_components).astype(np.float32)
        self.item_embeddings = np.random.rand(n_items, self.no_components).astype(np.float32)
        
    def recommend(
        self, 
        user_id: Union[int, str], 
        user_items: sp.csr_matrix, 
        n_items: int = 10, 
        **kwargs: Any
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
        # Convert string user_id to int if needed
        if isinstance(user_id, str):
            user_id = int(user_id)
        
        # Get number of items
        if self.item_embeddings is None:
            raise ValueError("Model has not been trained, item_embeddings is None")
        n_items_total = self.item_embeddings.shape[0]
        
        # Just return random recommendations
        top_items = np.random.choice(n_items_total, size=n_items, replace=False)
        top_scores = np.random.random(n_items)
        
        return top_items, top_scores
    
    def get_item_factors(self) -> np.ndarray:
        """Get item factors matrix.
        
        Returns:
            Item factors matrix
        """
        if self.item_embeddings is None:
            return np.array([])  # Return empty array if not initialized
        return np.asarray(self.item_embeddings)
    
    def save_model(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model parameters
        model_state = {
            "no_components": self.no_components,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "item_alpha": self.item_alpha,
            "user_alpha": self.user_alpha,
            "epochs": self.epochs,
            "user_biases": self.user_biases,
            "item_biases": self.item_biases,
            "user_embeddings": self.user_embeddings,
            "item_embeddings": self.item_embeddings,
        }
        
        with open(os.path.join(path, "lightfm_model.pkl"), "wb") as f:
            pickle.dump(model_state, f)
    
    def load_model(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        # Load model parameters
        with open(os.path.join(path, "lightfm_model.pkl"), "rb") as f:
            model_state = pickle.load(f)
        
        # Set model attributes
        self.no_components = model_state["no_components"]
        self.learning_rate = model_state["learning_rate"]
        self.loss = model_state["loss"]
        self.item_alpha = model_state["item_alpha"]
        self.user_alpha = model_state["user_alpha"]
        self.epochs = model_state["epochs"]
        self.user_biases = model_state["user_biases"]
        self.item_biases = model_state["item_biases"]
        self.user_embeddings = model_state["user_embeddings"]
        self.item_embeddings = model_state["item_embeddings"]