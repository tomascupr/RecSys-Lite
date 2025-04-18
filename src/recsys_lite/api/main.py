"""FastAPI service for RecSys-Lite."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from recsys_lite.indexing import FaissIndexBuilder
from recsys_lite.api.services import RecommendationService


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

    # Global variables / service container
    user_mapping = {}
    item_mapping = {}
    reverse_user_mapping = {}
    reverse_item_mapping = {}
    item_data = {}
    faiss_index = None
    model = None
    model_type = None
    user_item_matrix = None
    # Recommendation service (initialized on startup)
    rec_service = None

    @app.on_event("startup")
    async def startup_event() -> None:
        """Load model artifacts on startup."""
        nonlocal user_mapping, item_mapping, reverse_user_mapping, reverse_item_mapping, rec_service
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

            # Determine model type from directory name (lower‑cased).
            model_type = model_dir.name.lower()

            try:
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
                    # Unknown directory – proceed without a model.
                    model = None
            except FileNotFoundError:
                # Model artefact not present – acceptable in lightweight tests.
                model = None

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

            # Initialize recommendation service
            rec_service = RecommendationService(
                model=model,
                faiss_index=faiss_index,
                model_type=model_type,
                user_mapping=user_mapping,
                item_mapping=item_mapping,
                reverse_item_mapping=reverse_item_mapping,
                user_item_matrix=user_item_matrix,
            )
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
        """Get recommendations for a user."""
        # Update request count
        nonlocal request_count, recommendation_count, error_count
        request_count += 1

        # Ensure service is initialized
        if rec_service is None:
            raise HTTPException(status_code=503, detail="Recommender system not initialized")

        try:
            # Delegate to service layer
            item_ids, scores, item_meta = rec_service.recommend_for_user(
                user_id=user_id, k=k, use_faiss=use_faiss, item_data=item_data
            )
            # Build Pydantic response objects
            recommendations = [
                Recommendation(
                    item_id=i,
                    score=float(s),
                    title=m.get("title"),
                    category=m.get("category"),
                    brand=m.get("brand"),
                    price=m.get("price"),
                    img_url=m.get("img_url"),
                )
                for i, s, m in zip(item_ids, scores, item_meta)
            ]
            recommendation_count += len(recommendations)
            return RecommendationResponse(user_id=user_id, recommendations=recommendations)
        except HTTPException:
            # Propagate known HTTP errors
            raise
        except Exception as e:
            # Record and wrap unexpected errors
            error_count += 1
            print(f"Error generating recommendations: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating recommendations: {str(e)}",
            ) from e

    @app.get("/similar-items")
    async def similar_items(
        item_id: str = Query(..., description="Item ID to find similar items for"),
        k: int = Query(10, description="Number of similar items to return"),
    ) -> List[Recommendation]:
        """Get similar items."""
        # Update request count
        nonlocal request_count, recommendation_count, error_count
        request_count += 1

        # Ensure service is initialized
        if rec_service is None:
            raise HTTPException(status_code=503, detail="Recommender system not initialized")

        try:
            # Delegate to service layer
            item_ids, scores, item_meta = rec_service.find_similar_items(
                item_id=item_id, k=k, item_data=item_data
            )
            # Build Pydantic response objects
            recommendations = [
                Recommendation(
                    item_id=i,
                    score=float(s),
                    title=m.get("title"),
                    category=m.get("category"),
                    brand=m.get("brand"),
                    price=m.get("price"),
                    img_url=m.get("img_url"),
                )
                for i, s, m in zip(item_ids, scores, item_meta)
            ]
            recommendation_count += len(recommendations)
            return recommendations
        except HTTPException:
            # Propagate known HTTP errors
            raise
        except Exception as e:
            # Record and wrap unexpected errors
            error_count += 1
            print(f"Error finding similar items: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error finding similar items: {str(e)}",
            ) from e

    return app

    # ------------------------------------------------------------------
    # Run the *startup_event* coroutine once so that the application can be
    # used without entering the lifespan context (e.g. when instantiated by
    # the FastAPI TestClient outside of a *with* block).
    # ------------------------------------------------------------------

    import asyncio

    async def _startup_wrapper() -> None:  # pragma: no cover
        try:
            await startup_event()  # pylint: disable=not-callable
        except Exception as exc:  # noqa: BLE001
            # Swallow the exception – endpoints will handle missing artefacts.
            print(f"[RecSys‑Lite] Eager startup initialisation failed: {exc}")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside an existing event‑loop (unlikely here) – schedule
            # asynchronously so that it doesn't block.
            asyncio.ensure_future(_startup_wrapper())
        else:
            loop.run_until_complete(_startup_wrapper())
    except RuntimeError:
        # No loop in this thread.
        asyncio.run(_startup_wrapper())


# For backwards compatibility
app = create_app()
