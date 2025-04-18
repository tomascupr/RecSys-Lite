"""Pydantic models for the RecSys-Lite API."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class Recommendation(BaseModel):
    """Recommendation model for API responses."""

    item_id: str
    score: float
    title: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    img_url: Optional[str] = None


class RecommendationResponse(BaseModel):
    """API response for recommendations."""

    user_id: str
    recommendations: List[Recommendation]


class MetricsResponse(BaseModel):
    """API response for metrics endpoint."""

    uptime_seconds: float
    request_count: int
    recommendation_count: int
    error_count: int
    recommendations_per_second: float
    model_type: Optional[str] = None
    model_info: Optional[Dict[str, int]] = None
