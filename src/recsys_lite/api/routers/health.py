"""Router for health and monitoring endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, Depends

from recsys_lite.api.dependencies import get_stats
from recsys_lite.api.models import MetricsResponse

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
) -> MetricsResponse:
    """Metrics endpoint for monitoring.

    Args:
        stats: Statistics from app state

    Returns:
        Metrics response
    """
    return MetricsResponse(**stats)
