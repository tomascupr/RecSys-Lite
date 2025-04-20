"""Dependencies for FastAPI application."""

from typing import Any, Callable, Dict, cast

from fastapi import Depends

from recsys_lite.api.services import RecommendationService
from recsys_lite.api.state import APIState

# Global state instance
_api_state = APIState()


def get_api_state() -> APIState:
    """Get API state.

    Returns:
        API state
    """
    return _api_state


def get_recommendation_service(state: APIState = Depends(get_api_state)) -> RecommendationService:
    """Get recommendation service.

    Args:
        state: API state

    Returns:
        Recommendation service

    Raises:
        RuntimeError: If recommendation service is not initialized
    """
    if not hasattr(state, "recommendation_service") or state.recommendation_service is None:
        raise RuntimeError("Recommendation service not initialized")
    return cast(RecommendationService, state.recommendation_service)


def get_stats(state: APIState = Depends(get_api_state)) -> Dict[str, Any]:
    """Get API statistics.

    Args:
        state: API state

    Returns:
        API statistics
    """
    return state.get_metrics()


def increment_request_counter(state: APIState = Depends(get_api_state)) -> None:
    """Increment request counter middleware.

    Args:
        state: API state
    """
    state.increase_request_count()


def get_error_handler() -> Callable[[Exception], None]:
    """Get error handler.

    Returns:
        Error handler function
    """

    def log_error(exc: Exception) -> None:
        """Log error and increment error counter.

        Args:
            exc: Exception
        """
        _api_state.increase_error_count()
        print(f"Error: {exc}")

    return log_error
