"""Error handling for RecSys-Lite API."""

import logging
from typing import Any, Callable, Dict, Optional, Type, TypedDict, cast

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("recsys-lite.api")


class RecSysError(Exception):
    """Base exception for RecSys-Lite API errors."""
    
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "An unexpected error occurred"
    
    def __init__(self, detail: Optional[str] = None):
        """Initialize exception.
        
        Args:
            detail: Error detail message
        """
        if detail:
            self.detail = detail
        super().__init__(self.detail)


class NotFoundError(RecSysError):
    """Error raised when a resource is not found."""
    
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Resource not found"


class UserNotFoundError(NotFoundError):
    """Error raised when a user is not found."""
    
    detail = "User not found"
    
    def __init__(self, user_id: Optional[str] = None):
        """Initialize exception.
        
        Args:
            user_id: User ID that was not found
        """
        detail = f"User {user_id} not found" if user_id else self.detail
        super().__init__(detail)


class ItemNotFoundError(NotFoundError):
    """Error raised when an item is not found."""
    
    detail = "Item not found"
    
    def __init__(self, item_id: Optional[str] = None):
        """Initialize exception.
        
        Args:
            item_id: Item ID that was not found
        """
        detail = f"Item {item_id} not found" if item_id else self.detail
        super().__init__(detail)


class ServiceUnavailableError(RecSysError):
    """Error raised when a service is unavailable."""
    
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Service unavailable"


class ModelNotInitializedError(ServiceUnavailableError):
    """Error raised when the model is not initialized."""
    
    detail = "Recommender system not initialized"


def add_error_handlers(app: FastAPI) -> None:
    """Add error handlers to FastAPI app.
    
    Args:
        app: FastAPI application
    """
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle validation errors.
        
        Args:
            request: FastAPI request
            exc: Validation exception
            
        Returns:
            JSON response with error details
        """
        logger.warning(f"Validation error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )
    
    @app.exception_handler(RecSysError)
    async def recsys_exception_handler(request: Request, exc: RecSysError) -> JSONResponse:
        """Handle RecSys-Lite API errors.
        
        Args:
            request: FastAPI request
            exc: RecSys-Lite API exception
            
        Returns:
            JSON response with error details
        """
        logger.error(f"API error: {exc}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle general exceptions.
        
        Args:
            request: FastAPI request
            exc: Exception
            
        Returns:
            JSON response with error details
        """
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred"},
        )


class ErrorResponse(TypedDict):
    """Type definition for error response."""
    
    status_code: int
    detail: str


def _create_error_response(error: RecSysError) -> ErrorResponse:
    """Create error response from RecSys error.
    
    Args:
        error: RecSys error
        
    Returns:
        Error response
    """
    return {"status_code": error.status_code, "detail": error.detail}


ERROR_TYPES: Dict[Type[Exception], Callable[[Exception], ErrorResponse]] = {
    UserNotFoundError: lambda e: _create_error_response(cast(RecSysError, e)),
    ItemNotFoundError: lambda e: _create_error_response(cast(RecSysError, e)),
    ModelNotInitializedError: lambda e: _create_error_response(cast(RecSysError, e)),
}