# Logging System

RecSys-Lite includes a centralized logging system that provides consistent logging across all components of the application.

## Features

- Centralized logging configuration
- Standardized log levels and formats
- Structured logging with context information
- Support for file and console logging
- Helper functions for common logging patterns

## Usage

### Basic Logging

```python
from recsys_lite.utils.logging import get_logger

# Get a logger for your module
logger = get_logger("my_module")

# Log messages at different levels
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

### Structured Logging with Context

```python
# Add context information to log messages
logger.info(
    "Generated recommendations for user",
    extra={
        "user_id": "user123",
        "recommendation_count": 10,
        "use_faiss": True
    }
)
```

### Exception Logging

```python
from recsys_lite.utils.logging import log_exception, LogLevel

try:
    # Some code that might raise an exception
    result = process_data()
except Exception as e:
    # Log the exception with context
    log_exception(
        logger,
        "Error processing data",
        e,
        level=LogLevel.ERROR,
        extra={"data_id": "data123"}
    )
```

### Configuring Logging

```python
from recsys_lite.utils.logging import configure_logging, LogLevel

# Configure logging with default settings (INFO level, console output)
configure_logging()

# Configure logging with custom settings
configure_logging(
    level=LogLevel.DEBUG,
    log_file="/path/to/logfile.log",
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

## Log Levels

The `LogLevel` enum provides standardized log levels:

- `LogLevel.DEBUG`: Detailed information for debugging
- `LogLevel.INFO`: General information about system operation
- `LogLevel.WARNING`: Indication of potential issues
- `LogLevel.ERROR`: Error conditions that might still allow the application to continue
- `LogLevel.CRITICAL`: Severe error conditions that might cause the application to terminate

## Best Practices

1. **Use the right log level**: Reserve ERROR and CRITICAL for actual errors, use INFO for normal operations, and DEBUG for detailed troubleshooting information.

2. **Add context**: Always include relevant context information using the `extra` parameter.

3. **Be consistent**: Use the same logging patterns across the codebase.

4. **Log exceptions properly**: Use `log_exception` to ensure consistent exception logging.

5. **Use structured logging**: Include structured data in the `extra` parameter rather than formatting it into the message string.