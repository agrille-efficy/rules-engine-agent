"""
Structured logging configuration using structlog.
Provides JSON logging, correlation IDs, and consistent log formatting.
"""
import os
import sys
import logging
import structlog
from typing import Optional
from pathlib import Path
from datetime import datetime
from pythonjsonlogger import jsonlogger


def setup_structured_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_logs: bool = True,
    service_name: str = "rules-engine-agent"
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_logs: Whether to use JSON format (True) or human-readable format (False)
        service_name: Name of the service for log identification
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Shared processors for all configurations
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_logs:
        # JSON logging for production/aggregation systems
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure JSON formatter for standard logging
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            json_formatter = jsonlogger.JsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s',
                rename_fields={
                    'levelname': 'level',
                    'asctime': 'timestamp'
                }
            )
            file_handler.setFormatter(json_formatter)
            logging.root.addHandler(file_handler)
    else:
        # Human-readable logging for development
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    # Add service metadata to all logs
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        service=service_name,
        environment=os.getenv("ENVIRONMENT", "development")
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class CorrelationIdContext:
    """Context manager for correlation ID tracking across requests."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize correlation ID context.
        
        Args:
            correlation_id: Optional correlation ID, generates one if not provided
        """
        import uuid
        self.correlation_id = correlation_id or f"corr-{uuid.uuid4().hex[:16]}"
        self._previous_context = None
    
    def __enter__(self):
        """Bind correlation ID to context."""
        self._previous_context = structlog.contextvars.get_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=self.correlation_id)
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous context."""
        structlog.contextvars.clear_contextvars()
        if self._previous_context:
            for key, value in self._previous_context.items():
                structlog.contextvars.bind_contextvars(**{key: value})


def log_function_call(logger: structlog.stdlib.BoundLogger):
    """
    Decorator to log function entry/exit with timing.
    
    Args:
        logger: Structured logger instance
        
    Example:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            pass
    """
    def decorator(func):
        from functools import wraps
        import time
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Sanitize arguments for logging
            safe_args = [str(arg)[:100] for arg in args]
            safe_kwargs = {k: str(v)[:100] for k, v in kwargs.items()}
            
            logger.debug(
                "function_call_start",
                function=func.__name__,
                args=safe_args,
                kwargs=safe_kwargs
            )
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.debug(
                    "function_call_success",
                    function=func.__name__,
                    duration_seconds=round(duration, 3)
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    "function_call_error",
                    function=func.__name__,
                    duration_seconds=round(duration, 3),
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


def log_api_call(logger: structlog.stdlib.BoundLogger, api_name: str):
    """
    Decorator to log external API calls with timing and error handling.
    
    Args:
        logger: Structured logger instance
        api_name: Name of the API (e.g., "openai", "qdrant")
        
    Example:
        @log_api_call(logger, "openai")
        def call_openai_api():
            pass
    """
    def decorator(func):
        from functools import wraps
        import time
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            logger.info(
                "api_call_start",
                api=api_name,
                function=func.__name__
            )
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    "api_call_success",
                    api=api_name,
                    function=func.__name__,
                    duration_seconds=round(duration, 3)
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    "api_call_error",
                    api=api_name,
                    function=func.__name__,
                    duration_seconds=round(duration, 3),
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator
