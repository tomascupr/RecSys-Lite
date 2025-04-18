"""Service layer for RecSys-Lite API."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from fastapi import HTTPException

from recsys_lite.models.base import BaseRecommender
from recsys_lite.indexing import FaissIndexBuilder


class VectorService:
    """Service for retrieving vector representations from different model types."""
    
    @staticmethod
    def get_user_vector(
        model: BaseRecommender, 
        user_idx: int, 
        model_type: str,
        vector_size: Optional[int] = None
    ) -> np.ndarray:
        """Get user vector based on model type.
        
        Args:
            model: Recommendation model
            user_idx: User index
            model_type: Model type string
            vector_size: Vector size for fallback random vector
            
        Returns:
            User vector as numpy array
        """
        # For matrix factorization models
        if hasattr(model, "get_user_factors"):
            user_factors = model.get_user_factors()
            if user_factors is not None and user_idx < len(user_factors):
                return user_factors[user_idx].reshape(1, -1).astype(np.float32)
        
        # For LightFM
        if hasattr(model, "get_user_representations"):
            _, user_vectors = model.get_user_representations()
            if user_vectors is not None and user_idx < len(user_vectors):
                return user_vectors[user_idx].reshape(1, -1).astype(np.float32)
        
        # For other embeddings models
        if hasattr(model, "get_user_embedding") and hasattr(model, "user_embeddings"):
            if model.user_embeddings is not None and user_idx in model.user_embeddings:
                return np.array(model.user_embeddings[user_idx]).reshape(1, -1).astype(np.float32)
        
        # Fallback to random vector
        if vector_size is None:
            vector_size = getattr(model, "factors", 100)
        return np.random.random(vector_size).astype(np.float32).reshape(1, -1)
    
    @staticmethod
    def get_item_vector(
        model: BaseRecommender, 
        item_idx: int, 
        item_id: str,
        model_type: str,
        vector_size: Optional[int] = None
    ) -> np.ndarray:
        """Get item vector based on model type.
        
        Args:
            model: Recommendation model
            item_idx: Item index
            item_id: Item ID string
            model_type: Model type string
            vector_size: Vector size for fallback random vector
            
        Returns:
            Item vector as numpy array
        """
        # For matrix factorization models
        if hasattr(model, "get_item_factors"):
            item_factors = model.get_item_factors()
            if item_factors is not None and item_idx < len(item_factors):
                return item_factors[item_idx].reshape(1, -1).astype(np.float32)
        
        # For LightFM
        if hasattr(model, "get_item_representations"):
            _, item_vectors = model.get_item_representations()
            if item_vectors is not None and item_idx < len(item_vectors):
                return item_vectors[item_idx].reshape(1, -1).astype(np.float32)
        
        # For Item2Vec
        if hasattr(model, "get_item_vectors") and hasattr(model, "item_vectors"):
            if model.item_vectors is not None and item_id in model.item_vectors:
                return np.array(model.item_vectors[item_id]).reshape(1, -1).astype(np.float32)
        
        # Fallback to random vector
        if vector_size is None:
            vector_size = getattr(model, "factors", 100)
        return np.random.random(vector_size).astype(np.float32).reshape(1, -1)


class RecommendationService:
    """Service for generating recommendations."""
    
    def __init__(
        self, 
        model: BaseRecommender,
        faiss_index: Any, 
        model_type: str, 
        user_mapping: Dict[str, int],
        item_mapping: Dict[str, int],
        reverse_item_mapping: Dict[int, str],
        user_item_matrix: Optional[sp.csr_matrix] = None
    ):
        """Initialize recommendation service.
        
        Args:
            model: Recommendation model
            faiss_index: FAISS index for similarity search
            model_type: Model type string
            user_mapping: Mapping from user IDs to indices
            item_mapping: Mapping from item IDs to indices
            reverse_item_mapping: Mapping from item indices to IDs
            user_item_matrix: User-item interaction matrix
        """
        self.model = model
        self.faiss_index = faiss_index
        self.model_type = model_type
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.reverse_item_mapping = reverse_item_mapping
        self.user_item_matrix = user_item_matrix
        self.vector_service = VectorService()
    
    def recommend_for_user(
        self, 
        user_id: str, 
        k: int = 10, 
        use_faiss: bool = True,
        item_data: Optional[Dict[str, Dict[str, any]]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, any]]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID
            k: Number of recommendations to return
            use_faiss: Whether to use FAISS index
            item_data: Item metadata
            
        Returns:
            Tuple of (item_ids, scores, item_metadata)
            
        Raises:
            HTTPException: If user ID not found or recommendation system not initialized
        """
        if not self.faiss_index:
            raise HTTPException(status_code=503, detail="Recommender system not initialized")
        
        if user_id not in self.user_mapping:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        user_idx = int(self.user_mapping[user_id])
        
        if use_faiss:
            # Get user vector
            user_vector = self.vector_service.get_user_vector(
                model=self.model,
                user_idx=user_idx,
                model_type=self.model_type,
                vector_size=self.faiss_index.d
            )
            
            # Search for similar items
            distances, indices = self.faiss_index.search(user_vector, k)
            
            # Process results
            item_ids = []
            scores = []
            
            for idx, score in zip(indices[0], distances[0]):
                if idx == -1:  # Faiss returns -1 for no results
                    continue
                
                # Get item ID from index
                item_id = self.reverse_item_mapping.get(int(idx), f"unknown_{idx}")
                item_ids.append(item_id)
                scores.append(float(score))
        else:
            # Use model's recommend method directly
            if self.user_item_matrix is None:
                # Fallback empty matrix
                import scipy.sparse as sp
                user_items = sp.csr_matrix((1, len(self.item_mapping)))
            else:
                if user_idx < self.user_item_matrix.shape[0]:
                    user_items = self.user_item_matrix[user_idx].reshape(1, -1)
                else:
                    user_items = sp.csr_matrix((1, self.user_item_matrix.shape[1]))
            
            # Get recommendations
            item_indices, scores = self.model.recommend(
                user_id=user_idx,
                user_items=user_items,
                n_items=k
            )
            
            # Convert item indices to IDs
            item_ids = [self.reverse_item_mapping.get(int(idx), f"unknown_{idx}") for idx in item_indices]
        
        # Get item metadata if available
        item_metadata = []
        if item_data:
            for item_id in item_ids:
                if item_id in item_data:
                    item_metadata.append(item_data[item_id])
                else:
                    item_metadata.append({})
        else:
            item_metadata = [{} for _ in item_ids]
        
        return item_ids, scores, item_metadata
    
    def find_similar_items(
        self, 
        item_id: str, 
        k: int = 10,
        item_data: Optional[Dict[str, Dict[str, any]]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, any]]]:
        """Find similar items.
        
        Args:
            item_id: Item ID
            k: Number of similar items to return
            item_data: Item metadata
            
        Returns:
            Tuple of (item_ids, scores, item_metadata)
            
        Raises:
            HTTPException: If item ID not found or recommender system not initialized
        """
        if not self.faiss_index:
            raise HTTPException(status_code=503, detail="Recommender system not initialized")
        
        if item_id not in self.item_mapping:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
        
        item_idx = int(self.item_mapping[item_id])
        
        # Get item vector
        item_vector = self.vector_service.get_item_vector(
            model=self.model,
            item_idx=item_idx,
            item_id=item_id,
            model_type=self.model_type,
            vector_size=self.faiss_index.d
        )
        
        # Search for similar items
        distances, indices = self.faiss_index.search(item_vector, k + 1)  # +1 to account for the item itself
        
        # Process results
        item_ids = []
        scores = []
        seen_items = set()  # To avoid duplicates
        
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:  # Faiss returns -1 for no results
                continue
            
            # Get item ID from index
            similar_item_id = self.reverse_item_mapping.get(int(idx), f"unknown_{idx}")
            
            # Skip the query item and avoid duplicates
            if similar_item_id == item_id or similar_item_id in seen_items:
                continue
            
            seen_items.add(similar_item_id)
            item_ids.append(similar_item_id)
            scores.append(float(score))
            
            # Stop once we have enough recommendations
            if len(item_ids) >= k:
                break
        
        # Get item metadata if available
        item_metadata = []
        if item_data:
            for similar_item_id in item_ids:
                if similar_item_id in item_data:
                    item_metadata.append(item_data[similar_item_id])
                else:
                    item_metadata.append({})
        else:
            item_metadata = [{} for _ in item_ids]
        
        return item_ids, scores, item_metadata