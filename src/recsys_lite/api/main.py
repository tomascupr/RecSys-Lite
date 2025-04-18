"""FastAPI service for RecSys-Lite."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    reverse_user_mapping = {}
    reverse_item_mapping = {}
    item_data = {}
    faiss_index = None
    model = None
    model_type = None
    user_item_matrix = None

    @app.on_event("startup")
    async def startup_event() -> None:
        """Load model artifacts on startup."""
        nonlocal user_mapping, item_mapping, reverse_user_mapping, reverse_item_mapping
        nonlocal faiss_index, item_data, model, model_type, user_item_matrix

        # Load user and item mappings
        try:
            with open(model_dir / "user_mapping.json", "r") as f:
                user_mapping = json.load(f)

            with open(model_dir / "item_mapping.json", "r") as f:
                item_mapping = json.load(f)

            # Create reverse mappings (for efficient lookup)
            reverse_user_mapping = {int(v): k for k, v in user_mapping.items()}
            reverse_item_mapping = {int(v): k for k, v in item_mapping.items()}

            # Determine model type from directory
            model_type = model_dir.name

            # Load the appropriate model based on type
            if model_type == "als":
                from recsys_lite.models import ALSModel

                model = ALSModel()
                model.load_model(str(model_dir / "als_model.pkl"))
            elif model_type == "bpr":
                from recsys_lite.models import BPRModel

                model = BPRModel()
                model.load_model(str(model_dir / "bpr_model.pkl"))
            elif model_type == "item2vec":
                from recsys_lite.models import Item2VecModel

                model = Item2VecModel()
                model.load_model(str(model_dir / "item2vec_model.pkl"))
            elif model_type == "lightfm":
                from recsys_lite.models import LightFMModel

                model = LightFMModel()
                model.load_model(str(model_dir / "lightfm_model.pkl"))
            elif model_type == "gru4rec":
                from recsys_lite.models import GRU4Rec

                model = GRU4Rec(n_items=len(item_mapping))
                model.load_model(str(model_dir / "gru4rec_model.pkl"))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Load Faiss index
            index_builder = FaissIndexBuilder.load(str(model_dir / "faiss_index"))
            faiss_index = index_builder.index

            # Load item data if available
            item_data_path = model_dir.parent.parent / "data" / "items.json"
            if item_data_path.exists():
                with open(item_data_path, "r") as f:
                    item_data = json.load(f)
            else:
                # Create empty dictionary if file doesn't exist
                item_data = {}

            # Create an empty user-item matrix for recommendations
            # This will be used to track items the user has already seen
            import scipy.sparse as sp

            user_item_matrix = sp.csr_matrix((len(user_mapping), len(item_mapping)))

            print(f"API initialized with model type: {model_type}")
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            raise

    # Track API usage for metrics
    request_count = 0
    recommendation_count = 0
    error_count = 0
    start_time = time.time()

    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/metrics")
    async def metrics() -> Dict[str, Any]:
        """Metrics endpoint for monitoring."""
        uptime = time.time() - start_time

        return {
            "uptime_seconds": round(uptime, 2),
            "request_count": request_count,
            "recommendation_count": recommendation_count,
            "error_count": error_count,
            "recommendations_per_second": round(recommendation_count / max(uptime, 1), 2),
            "model_type": model_type,
            "model_info": {
                "users": len(user_mapping) if user_mapping else 0,
                "items": len(item_mapping) if item_mapping else 0,
            },
        }

    @app.get("/recommend", response_model=RecommendationResponse)
    async def recommend(
        user_id: str = Query(..., description="User ID to get recommendations for"),
        k: int = Query(10, description="Number of recommendations to return"),
        use_faiss: bool = Query(
            True,
            description="Whether to use Faiss index or direct model recommendations",
        ),
    ) -> RecommendationResponse:
        """Get recommendations for a user.

        Args:
            user_id: User ID to get recommendations for
            k: Number of recommendations to return
            use_faiss: Whether to use Faiss index (faster) or direct model predictions

        Returns:
            Recommendation response
        """
        # Update metrics
        nonlocal request_count
        request_count += 1

        if not model or not faiss_index:
            raise HTTPException(status_code=503, detail="Recommender system not initialized")

        if user_id not in user_mapping:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        # Get user index
        user_idx = int(user_mapping[user_id])

        try:
            # Generate recommendations based on method
            if use_faiss:
                # For Faiss-based recommendation

                # Get user vector based on model type
                user_vector = None

                if model_type == "als" or model_type == "bpr":
                    # For matrix factorization models, get the user vector from user factors
                    if hasattr(model, "user_factors") and model.user_factors is not None:
                        user_vector = model.user_factors[user_idx].reshape(1, -1).astype(np.float32)
                elif model_type == "lightfm":
                    # For LightFM, get user representation
                    if hasattr(model, "get_user_representations"):
                        _, user_vector = model.get_user_representations()
                        user_vector = user_vector[user_idx].reshape(1, -1).astype(np.float32)
                elif model_type == "gru4rec":
                    # For GRU4Rec we'd typically need session data, not just a user ID
                    # As a fallback, we'll use a random vector
                    user_vector = np.random.random(faiss_index.d).astype(np.float32).reshape(1, -1)

                if user_vector is None:
                    # Fallback if we couldn't get a proper user vector
                    user_vector = np.random.random(faiss_index.d).astype(np.float32).reshape(1, -1)

                # Search for similar items using Faiss
                distances, indices = faiss_index.search(user_vector, k)

                # Process results
                item_ids = []
                scores = []

                for idx, score in zip(indices[0], distances[0], strict=True):
                    if idx == -1:  # Faiss returns -1 for no results
                        continue

                    # Get item ID from index using reverse mapping
                    item_id = reverse_item_mapping.get(int(idx), f"unknown_{idx}")
                    item_ids.append(item_id)
                    scores.append(float(score))
            else:
                # Direct model recommendations - more accurate but potentially slower
                # Each model implements the recommend method as part of BaseRecommender interface
                import scipy.sparse as sp

                # Create a sparse matrix for this user if needed
                if user_item_matrix is None:
                    # Fallback empty matrix
                    user_items = sp.csr_matrix((1, len(item_mapping)))
                else:
                    if user_idx < user_item_matrix.shape[0]:
                        user_items = user_item_matrix[user_idx].reshape(1, -1)
                    else:
                        user_items = sp.csr_matrix((1, user_item_matrix.shape[1]))

                # Get recommendations directly from model
                item_indices, scores = model.recommend(
                    user_id=user_idx, user_items=user_items, n_items=k
                )

                # Convert item indices to item IDs
                item_ids = []
                for idx in item_indices:
                    item_id = reverse_item_mapping.get(int(idx), f"unknown_{idx}")
                    item_ids.append(item_id)

            # Create recommendation objects
            recommendations = []
            for item_id, score in zip(item_ids, scores, strict=True):
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

            # Update metrics
            nonlocal recommendation_count
            recommendation_count += len(recommendations)

            return RecommendationResponse(
                user_id=user_id,
                recommendations=recommendations,
            )
        except Exception as e:
            # Update error metrics
            nonlocal error_count
            error_count += 1

            print(f"Error generating recommendations: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error generating recommendations: {str(e)}"
            ) from e

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
        # Update metrics
        nonlocal request_count
        request_count += 1

        if not faiss_index:
            raise HTTPException(status_code=503, detail="Recommender system not initialized")

        if item_id not in item_mapping:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

        try:
            # Get item vector based on model type
            item_vector = None
            item_idx = int(item_mapping[item_id])

            if model_type == "als" or model_type == "bpr":
                # For matrix factorization models, get the item vector from item factors
                if hasattr(model, "item_factors") and model.item_factors is not None:
                    item_vector = model.item_factors[item_idx].reshape(1, -1).astype(np.float32)
            elif model_type == "lightfm":
                # For LightFM, get item representation
                if hasattr(model, "get_item_representations"):
                    _, item_vector = model.get_item_representations()
                    item_vector = item_vector[item_idx].reshape(1, -1).astype(np.float32)
            elif model_type == "item2vec":
                # For Item2Vec, get the item embedding
                if hasattr(model, "get_item_vectors") and hasattr(model, "item_vectors"):
                    if model.item_vectors and item_id in model.item_vectors:
                        item_vector = (
                            np.array(model.item_vectors[item_id]).reshape(1, -1).astype(np.float32)
                        )

            if item_vector is None:
                # Fallback if we couldn't get a proper item vector
                item_vector = np.zeros((1, faiss_index.d), dtype=np.float32)
                # For demonstration, use a random vector with a high value for the item's own index
                # This will produce more meaningful results than an all-zeros vector
                item_vector = np.random.random((1, faiss_index.d)).astype(np.float32)

            # Search for similar items
            distances, indices = faiss_index.search(
                item_vector, k + 1
            )  # +1 because the item itself will be included

            # Create recommendations (try to skip the query item itself)
            recommendations = []
            seen_item_ids = set()  # Track items we've already included

            for score, idx in zip(distances[0], indices[0], strict=True):
                if idx == -1:  # Faiss returns -1 for no results
                    continue

                # Get item ID from index
                similar_item_id = reverse_item_mapping.get(int(idx), f"unknown_{idx}")

                # Skip the query item and avoid duplicates
                if similar_item_id == item_id or similar_item_id in seen_item_ids:
                    continue

                seen_item_ids.add(similar_item_id)

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

                # Stop once we have enough recommendations
                if len(recommendations) >= k:
                    break

            # Update metrics
            nonlocal recommendation_count
            recommendation_count += len(recommendations)

            return recommendations
        except Exception as e:
            # Update error metrics
            nonlocal error_count
            error_count += 1

            print(f"Error finding similar items: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error finding similar items: {str(e)}"
            ) from e

    return app


# For backwards compatibility
app = create_app()
