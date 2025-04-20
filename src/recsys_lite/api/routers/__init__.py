"""API routers for RecSys-Lite."""

from recsys_lite.api.routers.health import router as health_router
from recsys_lite.api.routers.recommendations import router as recommendation_router

__all__ = ["health_router", "recommendation_router"]
