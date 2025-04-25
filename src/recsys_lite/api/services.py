"""Service layer for RecSys-Lite API."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from recsys_lite.api.errors import (
    ItemNotFoundError, 
    ModelNotInitializedError, 
    UserNotFoundError,
    VectorRetrievalError
)
from recsys_lite.models.base import BaseRecommender, VectorProvider
from recsys_lite.utils.logging import get_logger, log_exception, LogLevel

logger = get_logger("api.services")


class EntityType(str, Enum):
    """Entity type enum for vector retrieval."""

    USER = "user"
    ITEM = "item"


class VectorService:
    """Service for retrieving vector representations from different model types."""

    def get_vector(
        self, 
        model: BaseRecommender, 
        entity_type: EntityType,
        entity_idx: int, 
        entity_id: Optional[str] = None,
        vector_size: Optional[int] = None
    ) -> NDArray[np.float32]:
        """Get vector representation for an entity (user or item).

        Args:
            model: Recommendation model
            entity_type: Type of entity (USER or ITEM)
            entity_idx: Entity index
            entity_id: Entity ID string (for error reporting)
            vector_size: Vector size for fallback random vector

        Returns:
            Entity vector as numpy array

        Raises:
            VectorRetrievalError: If vector retrieval fails
        """
        try:
            # Determine which methods to call based on entity type
            if entity_type == EntityType.USER:
                vector_method = "get_user_vectors"
                factors_method = "get_user_factors"
                entity_indices = [entity_idx]
            else:  # EntityType.ITEM
                vector_method = "get_item_vectors"
                factors_method = "get_item_factors"
                entity_indices = [entity_idx]

            # Try using the standardized interface first
            if hasattr(model, vector_method):
                vector_provider = cast(VectorProvider, model)
                try:
                    method = getattr(vector_provider, vector_method)
                    vectors = method(entity_indices)
                    
                    # Only accept numpy arrays from providers
                    if isinstance(vectors, np.ndarray) and vectors.size > 0:
                        return cast(NDArray[np.float32], vectors[0].reshape(1, -1).astype(np.float32))
                except Exception as e:
                    log_exception(
                        logger,
                        f"Failed to get {entity_type.value} vector using {vector_method}",
                        e,
                        level=LogLevel.WARNING,
                        extra={"entity_idx": entity_idx, "entity_id": entity_id}
                    )

            # Fallback for older model implementations
            if hasattr(model, factors_method):
                try:
                    factors_getter = getattr(model, factors_method)
                    factors = factors_getter()
                    if factors is not None and entity_idx < len(factors):
                        return cast(NDArray[np.float32], factors[entity_idx].reshape(1, -1).astype(np.float32))
                except Exception as e:
                    log_exception(
                        logger,
                        f"Failed to get {entity_type.value} vector using {factors_method}",
                        e,
                        level=LogLevel.WARNING,
                        extra={"entity_idx": entity_idx, "entity_id": entity_id}
                    )

            # Fallback to random vector
            if vector_size is None:
                vector_size = getattr(model, "factors", 100)
                
            logger.info(
                f"Using random vector for {entity_type.value} {entity_id or entity_idx}",
                extra={"vector_size": vector_size}
            )
            return cast(NDArray[np.float32], np.random.random(vector_size).astype(np.float32).reshape(1, -1))
            
        except Exception as e:
            # Catch any unexpected errors
            error_msg = f"Unexpected error retrieving {entity_type.value} vector"
            log_exception(logger, error_msg, e, level=LogLevel.ERROR)
            raise VectorRetrievalError(
                entity_type=entity_type.value,
                entity_id=str(entity_id) if entity_id else str(entity_idx),
                reason=str(e)
            )
    
    def get_user_vector(
        self, model: BaseRecommender, user_idx: int, vector_size: Optional[int] = None
    ) -> NDArray[np.float32]:
        """Get user vector from model.

        Args:
            model: Recommendation model
            user_idx: User index
            vector_size: Vector size for fallback random vector

        Returns:
            User vector as numpy array
        """
        return self.get_vector(
            model=model,
            entity_type=EntityType.USER,
            entity_idx=user_idx,
            vector_size=vector_size
        )

    def get_item_vector(
        self, model: BaseRecommender, item_idx: int, item_id: Optional[str] = None, vector_size: Optional[int] = None
    ) -> NDArray[np.float32]:
        """Get item vector from model.

        Args:
            model: Recommendation model
            item_idx: Item index
            item_id: Item ID string (for error reporting)
            vector_size: Vector size for fallback random vector

        Returns:
            Item vector as numpy array
        """
        return self.get_vector(
            model=model,
            entity_type=EntityType.ITEM,
            entity_idx=item_idx,
            entity_id=item_id,
            vector_size=vector_size
        )


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
        user_item_matrix: Optional[sp.csr_matrix] = None,
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
        item_data: Optional[Dict[str, Dict[str, Any]]] = None,
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
        self, user_idx: int, k: int, item_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Get recommendations using FAISS index.

        Args:
            user_idx: User index
            k: Number of recommendations
            item_data: Item metadata

        Returns:
            Tuple of (item_ids, scores, item_metadata)
        """
        try:
            # Get user vector
            user_vector = self.vector_service.get_vector(
                model=self.model, 
                entity_type=EntityType.USER,
                entity_idx=user_idx, 
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
            
            logger.debug(
                f"Generated {len(item_ids)} recommendations for user {user_idx} using Faiss",
                extra={"user_idx": user_idx, "recommendation_count": len(item_ids)}
            )
            
            return item_ids, scores_list, self._get_item_metadata(item_ids, item_data)
            
        except VectorRetrievalError:
            # Re-raise vector retrieval errors
            raise
        except Exception as e:
            # Log and wrap other errors
            log_exception(
                logger,
                "Error generating Faiss recommendations",
                e,
                level=LogLevel.ERROR,
                extra={"user_idx": user_idx, "k": k}
            )
            raise

    def _get_direct_recommendations(
        self, user_idx: int, k: int, item_data: Optional[Dict[str, Dict[str, Any]]] = None
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
        item_indices, scores = self.model.recommend(user_id=user_idx, user_items=user_items, n_items=k)

        # Convert item indices to IDs
        item_ids = [self.reverse_item_mapping.get(int(idx), f"unknown_{idx}") for idx in item_indices]

        scores_list = [float(score) for score in scores]
        return item_ids, scores_list, self._get_item_metadata(item_ids, item_data)

    def _get_item_metadata(
        self, item_ids: List[str], item_data: Optional[Dict[str, Dict[str, Any]]] = None
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
    
    def filter_recommendations(
        self, 
        item_ids: List[str], 
        scores: List[float], 
        item_metadata: List[Dict[str, Any]],
        categories: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_items: Optional[List[str]] = None,
        include_items: Optional[List[str]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]], Dict[str, Any]]:
        """Filter recommendations based on provided criteria.
        
        Args:
            item_ids: List of item IDs
            scores: List of scores
            item_metadata: List of item metadata dictionaries
            categories: List of categories to include
            brands: List of brands to include
            min_price: Minimum price
            max_price: Maximum price
            exclude_items: Item IDs to exclude
            include_items: Limit to these item IDs
            
        Returns:
            Tuple of (filtered_item_ids, filtered_scores, filtered_item_metadata, filter_info)
        """
        filtered_items = []
        filtered_scores = []
        filtered_metadata = []
        
        # Track filter metrics
        filter_info = {
            "original_count": len(item_ids),
            "filtered_count": 0,
            "filters_applied": {}
        }
        
        # Create exclusion set for fast lookups
        exclusion_set = set(exclude_items or [])
        
        # Create inclusion set if provided
        inclusion_set = set(include_items or [])
        has_inclusion_filter = bool(inclusion_set)
        
        # Track which filters were applied
        if categories:
            filter_info["filters_applied"]["categories"] = categories
        if brands:
            filter_info["filters_applied"]["brands"] = brands
        if min_price is not None:
            filter_info["filters_applied"]["min_price"] = min_price
        if max_price is not None:
            filter_info["filters_applied"]["max_price"] = max_price
        if exclude_items:
            filter_info["filters_applied"]["excluded_items"] = len(exclude_items)
        if include_items:
            filter_info["filters_applied"]["included_items"] = len(include_items)
        
        for item_id, score, metadata in zip(item_ids, scores, item_metadata, strict=False):
            # Skip excluded items
            if item_id in exclusion_set:
                continue
                
            # Skip if not in inclusion set (when provided)
            if has_inclusion_filter and item_id not in inclusion_set:
                continue
                
            # Filter by category
            if categories and metadata.get("category") not in categories:
                continue
                
            # Filter by brand
            if brands and metadata.get("brand") not in brands:
                continue
                
            # Filter by price
            price = metadata.get("price")
            if price is not None:
                if min_price is not None and price < min_price:
                    continue
                if max_price is not None and price > max_price:
                    continue
            elif min_price is not None or max_price is not None:
                # If we're filtering by price but this item has no price, skip it
                continue
                
            # Item passed all filters
            filtered_items.append(item_id)
            filtered_scores.append(score)
            filtered_metadata.append(metadata)
        
        filter_info["filtered_count"] = len(filtered_items)
        return filtered_items, filtered_scores, filtered_metadata, filter_info
        
    def paginate_results(
        self,
        item_ids: List[str],
        scores: List[float],
        item_metadata: List[Dict[str, Any]],
        page: int = 1,
        page_size: int = 10
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]], Dict[str, Any]]:
        """Paginate recommendations.
        
        Args:
            item_ids: List of item IDs
            scores: List of scores
            item_metadata: List of item metadata dictionaries
            page: Page number (1-based)
            page_size: Number of items per page
            
        Returns:
            Tuple of (paginated_item_ids, paginated_scores, paginated_item_metadata, pagination_info)
        """
        total_items = len(item_ids)
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        
        # Ensure page is within valid range
        page = max(1, min(page, total_pages))
        
        # Calculate slice indices
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)
        
        pagination_info = {
            "total": total_items,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
        
        # Return paginated results
        return (
            item_ids[start_idx:end_idx],
            scores[start_idx:end_idx],
            item_metadata[start_idx:end_idx],
            pagination_info
        )

    def find_similar_items(
        self, item_id: str, k: int = 10, item_data: Optional[Dict[str, Dict[str, Any]]] = None
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
            VectorRetrievalError: If vector retrieval fails
        """
        if not self.faiss_index:
            raise ModelNotInitializedError()

        if item_id not in self.item_mapping:
            raise ItemNotFoundError(item_id)

        item_idx = int(self.item_mapping[item_id])

        try:
            # Get item vector
            item_vector = self.vector_service.get_vector(
                model=self.model, 
                entity_type=EntityType.ITEM,
                entity_idx=item_idx, 
                entity_id=item_id,
                vector_size=self.faiss_index.d
            )

            # Search for similar items
            # +1 to account for the item itself
            distances, indices = self.faiss_index.search(item_vector, k + 1)

            # Process results
            item_ids = []
            scores = []
            seen_items = set()  # To avoid duplicates
            
            logger.debug(
                f"Finding similar items for item {item_id}",
                extra={"item_id": item_id, "item_idx": item_idx, "k": k}
            )
            
        except VectorRetrievalError:
            # Re-raise vector retrieval errors
            raise
        except Exception as e:
            # Log and wrap other errors
            log_exception(
                logger,
                "Error finding similar items",
                e,
                level=LogLevel.ERROR,
                extra={"item_id": item_id, "item_idx": item_idx, "k": k}
            )
            raise

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
