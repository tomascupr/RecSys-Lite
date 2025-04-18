"""FastAPI service for RecSys-Lite."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from recsys_lite.indexing import FaissIndexBuilder


class Recommendation(BaseModel):
    """Recommendation response model."""
    
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


def create_app(model_dir: Union[str, Path] = "model_artifacts/als") -> FastAPI:
    """Create FastAPI application.
    
    Args:
        model_dir: Path to model artifacts
    
    Returns:
        FastAPI application
    """
    model_dir = Path(model_dir)
    
    app = FastAPI(
        title="RecSys-Lite API",
        description="Lightweight recommendation system for small e-commerce shops",
        version="0.1.0",
    )
    
    # Global variables
    user_mapping = {}
    item_mapping = {}
    item_data = {}
    faiss_index = None
    
    @app.on_event("startup")
    async def startup_event() -> None:
        """Load model artifacts on startup."""
        nonlocal user_mapping, item_mapping, faiss_index, item_data
        
        # Load user and item mappings
        try:
            with open(model_dir / "user_mapping.json", "r") as f:
                user_mapping = json.load(f)
            
            with open(model_dir / "item_mapping.json", "r") as f:
                item_mapping = json.load(f)
                
            # Create reverse mappings
            reverse_user_mapping = {int(v): k for k, v in user_mapping.items()}
            reverse_item_mapping = {int(v): k for k, v in item_mapping.items()}
            
            # Load Faiss index
            index_builder = FaissIndexBuilder.load(model_dir / "faiss_index")
            faiss_index = index_builder.index
            
            # Load item data if available
            item_data_path = model_dir.parent.parent / "data" / "items.json"
            if item_data_path.exists():
                with open(item_data_path, "r") as f:
                    item_data = json.load(f)
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            raise
    
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
            Recommendation response
        """
        if not faiss_index:
            raise HTTPException(status_code=503, detail="Recommender system not initialized")
        
        if user_id not in user_mapping:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        # Get user vector (implementation depends on model type)
        # For now, let's just use a random vector as a placeholder
        user_idx = user_mapping[user_id]
        user_vector = np.random.random(faiss_index.d).astype(np.float32)
        
        # Search for similar items
        distances, indices = faiss_index.search(user_vector.reshape(1, -1), k)
        
        # Create recommendations
        recommendations = []
        for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Faiss returns -1 for no results
                continue
                
            # Get item ID from index
            item_id = list(item_mapping.keys())[list(item_mapping.values()).index(idx)]
            
            # Get item data if available
            item_info = {}
            if item_id in item_data:
                item_info = item_data[item_id]
            
            # Create recommendation
            rec = Recommendation(
                item_id=item_id,
                score=float(score),
                title=item_info.get("title"),
                category=item_info.get("category"),
                brand=item_info.get("brand"),
                price=item_info.get("price"),
                img_url=item_info.get("img_url"),
            )
            recommendations.append(rec)
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
        )
    
    @app.get("/similar-items")
    async def similar_items(
        item_id: str = Query(..., description="Item ID to find similar items for"),
        k: int = Query(10, description="Number of similar items to return"),
    ) -> List[Recommendation]:
        """Get similar items.
        
        Args:
            item_id: Item ID to find similar items for
            k: Number of similar items to return
        
        Returns:
            List of similar items
        """
        if not faiss_index:
            raise HTTPException(status_code=503, detail="Recommender system not initialized")
        
        if item_id not in item_mapping:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
        
        # Get item vector
        item_idx = item_mapping[item_id]
        item_vector = np.zeros((1, faiss_index.d), dtype=np.float32)
        
        # This is a placeholder - in a real system, we'd get the actual item vector
        # For ALS/BPR/LightFM, we'd use the item factors
        # For Item2Vec, we'd use the item embedding
        
        # Search for similar items
        distances, indices = faiss_index.search(item_vector, k + 1)  # +1 because the item itself will be included
        
        # Create recommendations (skip the first result which is the item itself)
        recommendations = []
        for i, (score, idx) in enumerate(zip(distances[0][1:], indices[0][1:])):
            if idx == -1:  # Faiss returns -1 for no results
                continue
                
            # Get item ID from index
            similar_item_id = list(item_mapping.keys())[list(item_mapping.values()).index(idx)]
            
            # Get item data if available
            item_info = {}
            if similar_item_id in item_data:
                item_info = item_data[similar_item_id]
            
            # Create recommendation
            rec = Recommendation(
                item_id=similar_item_id,
                score=float(score),
                title=item_info.get("title"),
                category=item_info.get("category"),
                brand=item_info.get("brand"),
                price=item_info.get("price"),
                img_url=item_info.get("img_url"),
            )
            recommendations.append(rec)
        
        return recommendations
    
    return app


# For backwards compatibility
app = create_app()