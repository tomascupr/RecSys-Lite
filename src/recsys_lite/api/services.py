"""Service layer for RecSys-Lite API."""

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

from recsys_lite.api.errors import ItemNotFoundError, ModelNotInitializedError, UserNotFoundError
from recsys_lite.models.base import BaseRecommender, VectorProvider


class VectorService:
    """Service for retrieving vector representations from different model types."""
    
    def get_user_vector(
        self,
        model: BaseRecommender, 
        user_idx: int,
        vector_size: Optional[int] = None
    ) -> NDArray[np.float32]:
        """Get user vector from model.
        
        Args:
            model: Recommendation model
            user_idx: User index
            vector_size: Vector size for fallback random vector
            
        Returns:
            User vector as numpy array
        """
        if hasattr(model, "get_user_vectors"):
            # Use the standardized interface if available
            vector_provider = cast(VectorProvider, model)
            user_vectors = vector_provider.get_user_vectors([user_idx])
            if user_vectors.size > 0:
                return user_vectors[0].reshape(1, -1).astype(np.float32)
        
        # Fallback for older model implementations
        if hasattr(model, "get_user_factors"):
            user_factors = model.get_user_factors()
            if user_factors is not None and user_idx < len(user_factors):
                return user_factors[user_idx].reshape(1, -1).astype(np.float32)
        
        # Fallback to random vector
        if vector_size is None:
            vector_size = getattr(model, "factors", 100)
        return np.random.random(vector_size).astype(np.float32).reshape(1, -1)
    
    def get_item_vector(
        self,
        model: BaseRecommender, 
        item_idx: int,
        item_id: str,
        vector_size: Optional[int] = None
    ) -> NDArray[np.float32]:
        """Get item vector from model.
        
        Args:
            model: Recommendation model
            item_idx: Item index
            item_id: Item ID string
            vector_size: Vector size for fallback random vector
            
        Returns:
            Item vector as numpy array
        """
        if hasattr(model, "get_item_vectors"):
            # Use the standardized interface if available
            vector_provider = cast(VectorProvider, model)
            item_vectors = vector_provider.get_item_vectors([item_id])
            if item_vectors.size > 0:
                return item_vectors[0].reshape(1, -1).astype(np.float32)
        
        # Fallback for older model implementations
        if hasattr(model, "get_item_factors"):
            item_factors = model.get_item_factors()
            if item_factors is not None and item_idx < len(item_factors):
                return item_factors[item_idx].reshape(1, -1).astype(np.float32)
        
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
        item_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID
            k: Number of recommendations to return
            use_faiss: Whether to use FAISS index
            item_data: Item metadata
            
        Returns:
            Tuple of (item_ids, scores, item_metadata)
            
        Raises:
            ModelNotInitializedError: If recommender system is not initialized
            UserNotFoundError: If user ID is not found
        """
        if not self.faiss_index:
            raise ModelNotInitializedError()
        
        if user_id not in self.user_mapping:
            raise UserNotFoundError(user_id)
        
        user_idx = int(self.user_mapping[user_id])
        
        if use_faiss:
            return self._get_faiss_recommendations(user_idx, k, item_data)
        else:
            return self._get_direct_recommendations(user_idx, k, item_data)
    
    def _get_faiss_recommendations(
        self, 
        user_idx: int, 
        k: int,
        item_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Get recommendations using FAISS index.
        
        Args:
            user_idx: User index
            k: Number of recommendations
            item_data: Item metadata
            
        Returns:
            Tuple of (item_ids, scores, item_metadata)
        """
        # Get user vector
        user_vector = self.vector_service.get_user_vector(
            model=self.model,
            user_idx=user_idx,
            vector_size=self.faiss_index.d
        )
        
        # Search for similar items
        distances, indices = self.faiss_index.search(user_vector, k)
        
        # Process results
        item_ids = []
        scores = []
        
        for idx, score in zip(indices[0], distances[0], strict=False):
            if idx == -1:  # Faiss returns -1 for no results
                continue
            
            # Get item ID from index
            item_id = self.reverse_item_mapping.get(int(idx), f"unknown_{idx}")
            item_ids.append(item_id)
            scores.append(float(score))
        
        scores_list = [float(score) for score in scores]
        return item_ids, scores_list, self._get_item_metadata(item_ids, item_data)
    
    def _get_direct_recommendations(
        self, 
        user_idx: int, 
        k: int,
        item_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Get recommendations directly from model.
        
        Args:
            user_idx: User index
            k: Number of recommendations
            item_data: Item metadata
            
        Returns:
            Tuple of (item_ids, scores, item_metadata)
        """
        # Prepare user-item matrix
        if self.user_item_matrix is None:
            # Fallback empty matrix
            user_items = sp.csr_matrix((1, len(self.item_mapping)))
        else:
            if user_idx < self.user_item_matrix.shape[0]:
                user_items = self.user_item_matrix[user_idx].reshape(1, -1)
            else:
                user_items = sp.csr_matrix((1, self.user_item_matrix.shape[1]))
        
        # Get recommendations from model
        item_indices, scores = self.model.recommend(
            user_id=user_idx,
            user_items=user_items,
            n_items=k
        )
        
        # Convert item indices to IDs
        item_ids = [self.reverse_item_mapping.get(int(idx), f"unknown_{idx}") for idx in item_indices]
        
        scores_list = [float(score) for score in scores]
        return item_ids, scores_list, self._get_item_metadata(item_ids, item_data)
    
    def _get_item_metadata(
        self, 
        item_ids: List[str],
        item_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Get metadata for items.
        
        Args:
            item_ids: List of item IDs
            item_data: Item metadata dictionary
            
        Returns:
            List of item metadata dictionaries
        """
        if not item_data:
            return [{} for _ in item_ids]
        
        return [item_data.get(item_id, {}) for item_id in item_ids]
    
    def find_similar_items(
        self, 
        item_id: str, 
        k: int = 10,
        item_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Find similar items.
        
        Args:
            item_id: Item ID
            k: Number of similar items to return
            item_data: Item metadata
            
        Returns:
            Tuple of (item_ids, scores, item_metadata)
            
        Raises:
            ModelNotInitializedError: If recommender system is not initialized
            ItemNotFoundError: If item ID is not found
        """
        if not self.faiss_index:
            raise ModelNotInitializedError()
        
        if item_id not in self.item_mapping:
            raise ItemNotFoundError(item_id)
        
        item_idx = int(self.item_mapping[item_id])
        
        # Get item vector
        item_vector = self.vector_service.get_item_vector(
            model=self.model,
            item_idx=item_idx,
            item_id=item_id,
            vector_size=self.faiss_index.d
        )
        
        # Search for similar items
        distances, indices = self.faiss_index.search(item_vector, k + 1)  # +1 to account for the item itself
        
        # Process results
        item_ids = []
        scores = []
        seen_items = set()  # To avoid duplicates
        
        for idx, score in zip(indices[0], distances[0], strict=False):
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
        
        scores_list = [float(score) for score in scores]
        return item_ids, scores_list, self._get_item_metadata(item_ids, item_data)