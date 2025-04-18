"""Item2Vec model implementation using Gensim."""

from typing import Dict, Any, List, Tuple

import numpy as np
from gensim.models import Word2Vec


class Item2VecModel:
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
        self.item_vectors = None
    
    def fit(self, user_sessions: List[List[str]]) -> None:
        """Fit the Item2Vec model.
        
        Args:
            user_sessions: List of user sessions, each a list of item IDs
        """
        self.model.build_vocab(user_sessions)
        self.model.train(
            user_sessions,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )
        self._update_item_vectors()
    
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