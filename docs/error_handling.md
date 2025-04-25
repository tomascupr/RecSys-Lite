# Error Handling

RecSys-Lite includes a standardized error handling system that provides consistent error responses across all components of the application.

## Error Hierarchy

The error system is built around a hierarchy of exception classes:

- `RecSysError`: Base exception for all RecSys-Lite errors
  - `NotFoundError`: Base class for "not found" errors
    - `UserNotFoundError`: Raised when a user is not found
    - `ItemNotFoundError`: Raised when an item is not found
  - `ServiceUnavailableError`: Base class for service unavailability errors
    - `ModelNotInitializedError`: Raised when the recommendation model is not initialized
  - `VectorError`: Base class for vector-related errors
    - `VectorRetrievalError`: Raised when vector retrieval fails

## API Error Responses

When an error occurs in the API, it is converted to a standardized JSON response:

```json
{
  "detail": "Error message describing what went wrong"
}
```

The HTTP status code is set based on the error type:
- 404 Not Found: For `NotFoundError` and its subclasses
- 503 Service Unavailable: For `ServiceUnavailableError` and its subclasses
- 500 Internal Server Error: For `VectorError` and other errors

## Using the Error System

### Raising Errors

```python
from recsys_lite.api.errors import UserNotFoundError, VectorRetrievalError

# Raise a simple error
if user_id not in user_mapping:
    raise UserNotFoundError(user_id)

# Raise an error with detailed information
try:
    # Some code that might fail
    vector = get_vector(entity_id)
except Exception as e:
    raise VectorRetrievalError(
        entity_type="user",
        entity_id=user_id,
        reason=str(e)
    )
```

### Handling Errors

```python
from recsys_lite.api.errors import UserNotFoundError, ModelNotInitializedError, VectorRetrievalError
from recsys_lite.utils.logging import log_exception, LogLevel

try:
    # Code that might raise various errors
    recommendations = get_recommendations(user_id)
except (UserNotFoundError, ModelNotInitializedError, VectorRetrievalError):
    # Known errors - re-raise to be handled by exception handlers
    raise
except Exception as e:
    # Unexpected errors
    log_exception(
        logger,
        "Unexpected error getting recommendations",
        e,
        level=LogLevel.ERROR,
        extra={"user_id": user_id}
    )
    # Optionally wrap in a more specific error
    raise RecSysError(f"Failed to get recommendations: {e}")
```

## Best Practices

1. **Use specific error types**: Choose the most specific error type for the situation.

2. **Include context**: Add relevant information to error messages to help with debugging.

3. **Consistent error handling**: Follow the pattern of catching specific errors first, then handling unexpected errors.

4. **Log before raising**: Always log errors before raising them to ensure they're captured.

5. **Clean error messages**: Make error messages clear and actionable for API consumers.