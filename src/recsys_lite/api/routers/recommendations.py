"""Router for recommendation endpoints."""

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from recsys_lite.api.models import Recommendation, RecommendationResponse
from recsys_lite.api.services import RecommendationService


# A function that will be a dependency to get the recommendation service
def get_recommendation_service():
    """Get recommendation service from app state.
    
    This will be injected from the main app through dependency_overrides.
    """
    raise NotImplementedError("Recommendation service must be provided by app")


router = APIRouter()


@router.get("/recommend", response_model=RecommendationResponse)
async def recommend(
    user_id: str = Query(..., description="User ID to get recommendations for"),
    k: int = Query(10, description="Number of recommendations to return"),
    use_faiss: bool = Query(True, description="Whether to use Faiss index or direct model"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    """Get recommendations for a user.
    
    Args:
        user_id: User ID to get recommendations for
        k: Number of recommendations to return
        use_faiss: Whether to use Faiss index (faster) or direct model predictions
        recommendation_service: Recommendation service from dependency injection
        
    Returns:
        Recommendation response
    """
    try:
        item_ids, scores, item_metadata = recommendation_service.recommend_for_user(
            user_id=user_id,
            k=k,
            use_faiss=use_faiss
        )
        
        # Create recommendation objects
        recommendations = []
        for item_id, score, metadata in zip(item_ids, scores, item_metadata):
            # Create recommendation
            rec = Recommendation(
                item_id=item_id,
                score=float(score),
                title=metadata.get("title"),
                category=metadata.get("category"),
                brand=metadata.get("brand"),
                price=metadata.get("price"),
                img_url=metadata.get("img_url"),
            )
            recommendations.append(rec)
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error and raise HTTP exception
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@router.get("/similar-items", response_model=List[Recommendation])
async def similar_items(
    item_id: str = Query(..., description="Item ID to find similar items for"),
    k: int = Query(10, description="Number of similar items to return"),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
) -> List[Recommendation]:
    """Get similar items.
    
    Args:
        item_id: Item ID to find similar items for
        k: Number of similar items to return
        recommendation_service: Recommendation service from dependency injection
        
    Returns:
        List of similar items
    """
    try:
        item_ids, scores, item_metadata = recommendation_service.find_similar_items(
            item_id=item_id,
            k=k
        )
        
        # Create recommendation objects
        recommendations = []
        for similar_item_id, score, metadata in zip(item_ids, scores, item_metadata):
            # Create recommendation
            rec = Recommendation(
                item_id=similar_item_id,
                score=float(score),
                title=metadata.get("title"),
                category=metadata.get("category"),
                brand=metadata.get("brand"),
                price=metadata.get("price"),
                img_url=metadata.get("img_url"),
            )
            recommendations.append(rec)
        
        return recommendations
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error and raise HTTP exception
        raise HTTPException(status_code=500, detail=f"Error finding similar items: {str(e)}")