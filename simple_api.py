"""Simple FastAPI service for RecSys-Lite."""

from typing import Dict, List

import duckdb
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
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
    # For demo purposes, we'll return random items from the database
    conn = duckdb.connect("data/recsys.db")

    # Check if user exists
    user_exists = conn.execute(
        f"SELECT COUNT(*) FROM events WHERE user_id = '{user_id}'"
    ).fetchone()[0]

    if not user_exists:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Get items not interacted with by the user
    items = conn.execute(
        f"""
        SELECT i.item_id, i.price 
        FROM items i
        WHERE i.item_id NOT IN (
            SELECT item_id FROM events WHERE user_id = '{user_id}'
        )
        ORDER BY RANDOM()
        LIMIT {k}
        """
    ).fetchall()

    conn.close()

    # Create recommendations with random scores
    recommendations = []
    for item_id, _price in items:
        score = np.random.random()  # Random score for demo
        recommendations.append(Recommendation(item_id=item_id, score=float(score)))

    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
