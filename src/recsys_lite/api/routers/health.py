"""Router for health and monitoring endpoints."""

import time
from typing import Dict, Any

from fastapi import APIRouter, Depends

from recsys_lite.api.models import MetricsResponse
from recsys_lite.api.services import RecommendationService


# A function that will be a dependency to get the stats
def get_stats():
    """Get statistics from app state.
    
    This will be injected from the main app through dependency_overrides.
    """
    raise NotImplementedError("Statistics must be provided by app")


# A function that will be a dependency to get the recommendation service
def get_recommendation_service():
    """Get recommendation service from app state.
    
    This will be injected from the main app through dependency_overrides.
    """
    raise NotImplementedError("Recommendation service must be provided by app")


router = APIRouter()


@router.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy"}


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(
    stats: Dict[str, Any] = Depends(get_stats),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
) -> MetricsResponse:
    """Metrics endpoint for monitoring.
    
    Args:
        stats: Statistics from app state
        recommendation_service: Recommendation service from dependency injection
        
    Returns:
        Metrics response
    """
    return MetricsResponse(
        uptime_seconds=stats.get("uptime_seconds", time.time() - stats.get("start_time", time.time())),
        request_count=stats.get("request_count", 0),
        recommendation_count=stats.get("recommendation_count", 0),
        error_count=stats.get("error_count", 0),
        recommendations_per_second=stats.get("recommendations_per_second", 0.0),
        model_type=recommendation_service.model_type if recommendation_service else None,
        model_info={
            "users": len(recommendation_service.user_mapping) if recommendation_service else 0,
            "items": len(recommendation_service.item_mapping) if recommendation_service else 0,
        },
    )