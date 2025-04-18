"""Item2Vec model implementation using Gensim."""

import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec

from recsys_lite.models.base import BaseRecommender


class Item2VecModel(BaseRecommender):
    """Item2Vec model for item embeddings."""
    
    def __init__(
        self,
        vector_size: int = 128,
        window: int = 5,
        min_count: int = 1,
        sg: int = 1,
        epochs: int = 5,
    ) -> None:
        """Initialize Item2Vec model.
        
        Args:
            vector_size: Dimensionality of embeddings
            window: Maximum distance between items in sequence
            min_count: Minimum item frequency
            sg: Training algorithm (1 for skip-gram, 0 for CBOW)
            epochs: Number of training epochs
        """
        self.model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            epochs=epochs,
            workers=0,  # Use all available cores
        )
        self.item_vectors: Dict[str, np.ndarray] = {}
        self.user_item_sessions: Dict[Union[int, str], List[str]] = {}
    
    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the Item2Vec model.
        
        Args:
            user_item_matrix: Sparse user-item interaction matrix
            **kwargs: Additional model-specific parameters
        """
        # Item2Vec normally works with sessions, not a matrix
        # For compatibility, extract sessions from the matrix if provided
        if "user_sessions" in kwargs:
            user_sessions = kwargs["user_sessions"]
        else:
            # Create simple sessions from the matrix (not ideal but works for compatibility)
            user_sessions = []
            for user_idx in range(user_item_matrix.shape[0]):
                items = user_item_matrix[user_idx].indices.astype(str).tolist()
                if items:
                    user_sessions.append(items)
        
        self.model.build_vocab(user_sessions)
        self.model.train(
            user_sessions,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )
        self._update_item_vectors()
        
        # Store sessions for later use
        for i, session in enumerate(user_sessions):
            self.user_item_sessions[i] = session
    
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
        # Get items that user has interacted with
        if isinstance(user_id, str):
            user_id_int = int(user_id)
        else:
            user_id_int = user_id
            
        # Get user's items from the matrix
        user_idx = user_id_int
        items_idx = user_items[user_idx].indices
        
        # Convert to strings to match the model's vocabulary
        items = [str(idx) for idx in items_idx]
        
        # Compute average item vector for user
        avg_vector = np.zeros(self.model.vector_size)
        count = 0
        for item in items:
            if item in self.item_vectors:
                avg_vector += self.item_vectors[item]
                count += 1
        
        if count > 0:
            avg_vector /= count
        
        # Find most similar items to user's average vector
        similar_items = []
        for item_id, vector in self.item_vectors.items():
            if item_id not in items:  # Skip items user already has
                similarity = np.dot(avg_vector, vector) / (np.linalg.norm(avg_vector) * np.linalg.norm(vector))
                similar_items.append((item_id, similarity))
        
        # Sort by similarity
        similar_items.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-n items
        top_items = similar_items[:n_items]
        
        # Return item IDs and scores
        item_ids = np.array([item_id for item_id, _ in top_items])
        scores = np.array([score for _, score in top_items])
        
        return item_ids, scores
    
    def _update_item_vectors(self) -> None:
        """Update item vectors from trained model."""
        items = list(self.model.wv.index_to_key)
        self.item_vectors = {item: self.model.wv[item] for item in items}
    
    def get_item_vectors(self) -> Dict[str, np.ndarray]:
        """Get item vectors dictionary.
        
        Returns:
            Dictionary mapping item IDs to embeddings
        """
        return self.item_vectors
    
    def get_item_vectors_matrix(self, item_ids: List[str]) -> np.ndarray:
        """Get item vectors as a matrix.
        
        Args:
            item_ids: List of item IDs to get vectors for
            
        Returns:
            Matrix of item embeddings
        """
        matrix = np.zeros((len(item_ids), self.model.vector_size))
        for i, item_id in enumerate(item_ids):
            if item_id in self.item_vectors:
                matrix[i] = self.item_vectors[item_id]
        return matrix
        
    def save_model(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save Word2Vec model
        self.model.save(os.path.join(path, "item2vec.model"))
        
        # Save item vectors separately for fast loading
        with open(os.path.join(path, "item_vectors.pkl"), "wb") as f:
            pickle.dump(self.item_vectors, f)
    
    def load_model(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        # Load Word2Vec model
        self.model = Word2Vec.load(os.path.join(path, "item2vec.model"))
        
        # Load item vectors
        with open(os.path.join(path, "item_vectors.pkl"), "rb") as f:
            self.item_vectors = pickle.load(f)