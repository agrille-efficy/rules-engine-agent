"""
Error handling and resilience patterns.
Provides retry logic, circuit breakers, and graceful degradation for external services.
"""
import time
import logging
import functools
from typing import Optional, Callable, Any, Type, Tuple, Dict
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)

from ..config.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class ServiceError(Exception):
    """Base exception for service-level errors."""
    pass


class OpenAIServiceError(ServiceError):
    """OpenAI API specific errors."""
    pass


class QdrantServiceError(ServiceError):
    """Qdrant service specific errors."""
    pass


class CircuitBreakerError(ServiceError):
    """Circuit breaker is open, service unavailable."""
    pass


class RetryExhaustedError(ServiceError):
    """All retry attempts exhausted."""
    pass


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures detected, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes needed to close from half-open
    timeout: int = 60  # Seconds before trying half-open
    expected_exceptions: Tuple[Type[Exception], ...] = (Exception,)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by failing fast when a service is down.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests blocked immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the service/circuit
            config: Optional configuration, uses defaults if not provided
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()
        
        logger.info(
            "circuit_breaker_initialized",
            circuit_name=name,
            failure_threshold=self.config.failure_threshold,
            timeout=self.config.timeout
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                logger.warning(
                    "circuit_breaker_blocked_call",
                    circuit_name=self.name,
                    state=self.state.value
                )
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. Service unavailable."
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exceptions as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        
        logger.debug(
            "circuit_breaker_success",
            circuit_name=self.name,
            state=self.state.value
        )
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        logger.warning(
            "circuit_breaker_failure",
            circuit_name=self.name,
            failure_count=self.failure_count,
            threshold=self.config.failure_threshold
        )
        
        if self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        
        elapsed = datetime.now() - self.last_failure_time
        return elapsed.total_seconds() >= self.config.timeout
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()
            
            logger.error(
                "circuit_breaker_opened",
                circuit_name=self.name,
                failure_count=self.failure_count
            )
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.last_state_change = datetime.now()
        
        logger.info(
            "circuit_breaker_half_open",
            circuit_name=self.name
        )
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = datetime.now()
        
        logger.info(
            "circuit_breaker_closed",
            circuit_name=self.name
        )
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        
        logger.info(
            "circuit_breaker_reset",
            circuit_name=self.name
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat()
        }


# ============================================================================
# Circuit Breaker Registry
# ============================================================================

class CircuitBreakerRegistry:
    """Global registry for circuit breakers."""
    
    _instance = None
    _breakers: Dict[str, CircuitBreaker] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_or_create(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers."""
        return {
            name: breaker.get_state()
            for name, breaker in self._breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


# ============================================================================
# Retry Decorators
# ============================================================================

def retry_on_api_error(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 10,
    service_name: str = "api"
):
    """
    Decorator to retry function on API errors with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        service_name: Name of the service for logging
        
    Example:
        @retry_on_api_error(max_attempts=3, service_name="openai")
        def call_openai():
            pass
    """
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type((
                ConnectionError,
                TimeoutError,
                OpenAIServiceError,
                QdrantServiceError,
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO),
            reraise=True
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(
                "retry_attempt_start",
                service=service_name,
                function=func.__name__
            )
            
            try:
                result = func(*args, **kwargs)
                
                logger.info(
                    "retry_attempt_success",
                    service=service_name,
                    function=func.__name__
                )
                
                return result
                
            except Exception as e:
                logger.error(
                    "retry_attempt_failed",
                    service=service_name,
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


def with_circuit_breaker(circuit_name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator to wrap function with circuit breaker.
    
    Args:
        circuit_name: Name of the circuit breaker
        config: Optional circuit breaker configuration
        
    Example:
        @with_circuit_breaker("openai")
        def call_openai():
            pass
    """
    def decorator(func):
        breaker = circuit_registry.get_or_create(circuit_name, config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def resilient_api_call(
    service_name: str,
    max_attempts: int = 3,
    circuit_config: Optional[CircuitBreakerConfig] = None
):
    """
    Combined decorator for retry logic + circuit breaker.
    
    Args:
        service_name: Name of the service
        max_attempts: Maximum retry attempts
        circuit_config: Optional circuit breaker configuration
        
    Example:
        @resilient_api_call("openai", max_attempts=3)
        def call_openai():
            pass
    """
    def decorator(func):
        # Apply circuit breaker first
        func = with_circuit_breaker(service_name, circuit_config)(func)
        # Then apply retry logic
        func = retry_on_api_error(max_attempts, service_name=service_name)(func)
        
        return func
    return decorator


# ============================================================================
# Graceful Degradation Helpers
# ============================================================================

class FallbackResult:
    """Result wrapper for fallback operations."""
    
    def __init__(
        self, 
        value: Any, 
        is_fallback: bool = False, 
        error: Optional[Exception] = None
    ):
        self.value = value
        self.is_fallback = is_fallback
        self.error = error


def with_fallback(fallback_value: Any, log_error: bool = True):
    """
    Decorator to provide fallback value on error.
    
    Args:
        fallback_value: Value to return on error
        log_error: Whether to log the error
        
    Example:
        @with_fallback(fallback_value=[])
        def get_data():
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return FallbackResult(value=result, is_fallback=False)
                
            except Exception as e:
                if log_error:
                    logger.warning(
                        "fallback_triggered",
                        function=func.__name__,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        fallback_value=str(fallback_value)[:100]
                    )
                
                return FallbackResult(
                    value=fallback_value,
                    is_fallback=True,
                    error=e
                )
        
        return wrapper
    return decorator


# ============================================================================
# Error Context Manager
# ============================================================================

class ErrorContext:
    """Context manager for error handling with structured logging."""
    
    def __init__(
        self,
        operation: str,
        raise_on_error: bool = True,
        log_level: str = "error"
    ):
        """
        Initialize error context.
        
        Args:
            operation: Name of the operation
            raise_on_error: Whether to re-raise exceptions
            log_level: Logging level for errors
        """
        self.operation = operation
        self.raise_on_error = raise_on_error
        self.log_level = log_level
        self.start_time = None
        self.error: Optional[Exception] = None
    
    def __enter__(self):
        """Enter context."""
        self.start_time = time.time()
        logger.debug("operation_start", operation=self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and handle errors."""
        duration = time.time() - self.start_time if self.start_time else 0
        
        if exc_type is not None:
            self.error = exc_val
            
            log_func = getattr(logger, self.log_level)
            log_func(
                "operation_failed",
                operation=self.operation,
                duration_seconds=round(duration, 3),
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )
            
            if not self.raise_on_error:
                return True  # Suppress exception
        else:
            logger.debug(
                "operation_success",
                operation=self.operation,
                duration_seconds=round(duration, 3)
            )
        
        return False
