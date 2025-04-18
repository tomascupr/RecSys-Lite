"""Router for recommendation endpoints."""

import logging
from typing import List

from fastapi import APIRouter, Depends, Query

from recsys_lite.api.dependencies import get_api_state, get_recommendation_service
from recsys_lite.api.errors import ItemNotFoundError, ModelNotInitializedError, UserNotFoundError
from recsys_lite.api.models import Recommendation, RecommendationResponse
from recsys_lite.api.services import RecommendationService
from recsys_lite.api.state import APIState

logger = logging.getLogger("recsys-lite.api")

router = APIRouter()


@router.get("/recommend", response_model=RecommendationResponse)
async def recommend(
    user_id: str = Query(..., description="User ID to get recommendations for"),
    k: int = Query(10, description="Number of recommendations to return"),
    use_faiss: bool = Query(True, description="Whether to use Faiss index or direct model"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    state: APIState = Depends(get_api_state),
) -> RecommendationResponse:
    """Get recommendations for a user.
    
    Args:
        user_id: User ID to get recommendations for
        k: Number of recommendations to return
        use_faiss: Whether to use Faiss index (faster) or direct model predictions
        recommendation_service: Recommendation service from dependency injection
        state: API state for metrics
        
    Returns:
        Recommendation response
    
    Raises:
        UserNotFoundError: If user ID is not found
        ModelNotInitializedError: If recommendation system is not initialized
    """
    try:
        # Get recommendations
        item_ids, scores, item_metadata = recommendation_service.recommend_for_user(
            user_id=user_id,
            k=k,
            use_faiss=use_faiss
        )
        
        # Create recommendation objects
        recommendations = [
            Recommendation(
                item_id=item_id,
                score=float(score),
                title=metadata.get("title"),
                category=metadata.get("category"),
                brand=metadata.get("brand"),
                price=metadata.get("price"),
                img_url=metadata.get("img_url"),
            )
            for item_id, score, metadata in zip(item_ids, scores, item_metadata, strict=False)
        ]
        
        # Update metrics
        state.increase_recommendation_count(len(recommendations))
        
        # Return response
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
        )
    except (UserNotFoundError, ModelNotInitializedError):
        # Known errors - re-raise to be handled by exception handlers
        raise
    except Exception as e:
        # Unexpected errors
        logger.exception(f"Error generating recommendations: {e}")
        state.increase_error_count()
        raise


@router.get("/similar-items", response_model=List[Recommendation])
async def similar_items(
    item_id: str = Query(..., description="Item ID to find similar items for"),
    k: int = Query(10, description="Number of similar items to return"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    state: APIState = Depends(get_api_state),
) -> List[Recommendation]:
    """Get similar items.
    
    Args:
        item_id: Item ID to find similar items for
        k: Number of similar items to return
        recommendation_service: Recommendation service from dependency injection
        state: API state for metrics
        
    Returns:
        List of similar items
    
    Raises:
        ItemNotFoundError: If item ID is not found
        ModelNotInitializedError: If recommendation system is not initialized
    """
    try:
        # Get similar items
        item_ids, scores, item_metadata = recommendation_service.find_similar_items(
            item_id=item_id,
            k=k
        )
        
        # Create recommendation objects
        recommendations = [
            Recommendation(
                item_id=similar_item_id,
                score=float(score),
                title=metadata.get("title"),
                category=metadata.get("category"),
                brand=metadata.get("brand"),
                price=metadata.get("price"),
                img_url=metadata.get("img_url"),
            )
            for similar_item_id, score, metadata in zip(
                item_ids, scores, item_metadata, strict=False)
        ]
        
        # Update metrics
        state.increase_recommendation_count(len(recommendations))
        
        # Return response
        return recommendations
    except (ItemNotFoundError, ModelNotInitializedError):
        # Known errors - re-raise to be handled by exception handlers
        raise
    except Exception as e:
        # Unexpected errors
        logger.exception(f"Error finding similar items: {e}")
        state.increase_error_count()
        raise