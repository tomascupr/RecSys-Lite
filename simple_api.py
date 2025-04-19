"""Simple FastAPI service for RecSys-Lite."""

import random
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(
    title="RecSys-Lite API",
    description="Lightweight recommendation system for small e-commerce shops",
    version="0.1.0",
)


class Recommendation(BaseModel):
    """Recommendation response model."""

    item_id: str
    score: float


class RecommendationResponse(BaseModel):
    """API response for recommendations."""

    user_id: str
    recommendations: List[Recommendation]


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/recommend", response_model=RecommendationResponse)
async def recommend(
    user_id: str = Query(..., description="User ID to get recommendations for"),
    k: int = Query(10, description="Number of recommendations to return"),
) -> RecommendationResponse:
    """Get recommendations for a user.

    Args:
        user_id: User ID to get recommendations for
        k: Number of recommendations to return

    Returns:
        Recommendation response with user ID and recommendations
    """
    # For demo purposes, we'll return random item IDs
    recommendations = []
    for _ in range(k):
        item_id = f"I_{random.randint(0, 500):04d}"
        score = random.random()
        recommendations.append(Recommendation(item_id=item_id, score=float(score)))

    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
