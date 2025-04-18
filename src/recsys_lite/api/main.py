"""FastAPI service for RecSys-Lite."""

import os
from typing import Dict, List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(
    title="RecSys-Lite API",
    description="Lightweight recommendation system for small e-commerce shops",
    version="0.1.0",
)

# Global variables for recommendation service
user_factors = None
item_factors = None
item_index = None
faiss_index = None
item_id_map = None


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
    if not faiss_index or user_id not in user_factors:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    # Get user factors
    user_vector = user_factors[user_id]
    
    # Search for similar items
    scores, indices = faiss_index.search(np.array([user_vector]), k)
    
    # Map indices to item IDs
    recommendations = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:  # Faiss uses -1 for no result
            item_id = item_id_map[idx]
            recommendations.append(Recommendation(item_id=item_id, score=float(score)))
    
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Load model artifacts on startup."""
    global user_factors, item_factors, faiss_index, item_id_map
    
    model_path = os.environ.get("MODEL_PATH", "../model_artifacts")
    
    # TODO: Load model artifacts
    # This will be implemented when model training is completed