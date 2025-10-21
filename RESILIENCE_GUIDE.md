# Error Handling & Resilience Features

## Overview

This document describes the comprehensive error handling, resilience, and structured logging features implemented in the Rules Engine Agent.

## Features Implemented

### 1. **Structured Logging with JSON Format**
- **JSON logging** for production environments (easy log aggregation)
- **Human-readable** logging for development
- **Correlation IDs** for request tracking across the system
- **Automatic context** binding (service name, environment)
- **Decorators** for function call and API call logging

### 2. **Retry Logic with Exponential Backoff**
- Automatic retry for transient failures
- Exponential backoff to avoid overwhelming services
- Configurable retry attempts and wait times
- Smart retry only on recoverable errors

### 3. **Circuit Breaker Pattern**
- Prevents cascading failures
- Three states: CLOSED (normal), OPEN (failing fast), HALF_OPEN (testing recovery)
- Configurable failure thresholds and timeouts
- Global circuit breaker registry

### 4. **Resilient API Clients**
- `ResilientOpenAIClient`: Wrapper for OpenAI API with built-in resilience
- `ResilientQdrantClient`: Wrapper for Qdrant with built-in resilience
- Combined retry + circuit breaker logic
- Structured logging for all API calls

### 5. **Health Check & Monitoring**
- Service health checks for OpenAI and Qdrant
- Circuit breaker status monitoring
- Overall system health reporting
- Response time tracking

---

## Quick Start

### Setup Structured Logging

```python
from Agent.config.logging_config import setup_structured_logging, get_logger, CorrelationIdContext

# Initialize logging (call once at startup)
setup_structured_logging(
    log_level="INFO",
    json_logs=True,  # Use JSON format for production
    log_file="logs/application.log",  # Optional file output
    service_name="rules-engine-agent"
)

# Get logger for your module
logger = get_logger(__name__)

# Use correlation IDs for request tracking
with CorrelationIdContext() as correlation_id:
    logger.info("processing_request", user_id=123, request_type="file_analysis")
    # All logs within this context will include the correlation_id
```

### Use Resilient API Clients

```python
from Agent.config import get_settings
from Agent.services.clients import ResilientOpenAIClient, ResilientQdrantClient

settings = get_settings()

# Initialize resilient clients
openai_client = ResilientOpenAIClient(
    api_key=settings.openai_api_key,
    model=settings.llm_model,
    temperature=settings.temperature,
    max_retries=3  # Will retry up to 3 times with exponential backoff
)

qdrant_client = ResilientQdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    max_retries=3
)

# Use clients - they handle retries and circuit breakers automatically
try:
    # Generate embedding with automatic retry
    embedding = openai_client.generate_embedding("sample text")
    
    # Search Qdrant with automatic retry
    results = qdrant_client.search(
        collection_name="my_collection",
        query_vector=embedding,
        limit=10
    )
    
except OpenAIServiceError as e:
    logger.error("openai_failed_after_retries", error=str(e))
    
except QdrantServiceError as e:
    logger.error("qdrant_failed_after_retries", error=str(e))
```

### Run Health Checks

```python
from Agent.services.health_check import health_service

# Check individual services
openai_health = health_service.check_openai(openai_client)
qdrant_health = health_service.check_qdrant(qdrant_client)

# Get overall system health
overall_health = health_service.get_overall_health()

# Print formatted report
health_service.print_health_report()
```

---

## Advanced Usage

### Custom Retry Logic

```python
from Agent.core.resilience import retry_on_api_error, with_circuit_breaker, resilient_api_call

# Method 1: Retry only
@retry_on_api_error(max_attempts=5, min_wait=2, max_wait=30, service_name="custom_api")
def call_custom_api():
    # Your API call here
    pass

# Method 2: Circuit breaker only
@with_circuit_breaker("custom_api")
def call_custom_api():
    pass

# Method 3: Combined (recommended)
@resilient_api_call("custom_api", max_attempts=3)
def call_custom_api():
    pass
```

### Custom Circuit Breaker Configuration

```python
from Agent.core.resilience import CircuitBreakerConfig, resilient_api_call

custom_config = CircuitBreakerConfig(
    failure_threshold=10,  # Open circuit after 10 failures
    success_threshold=3,   # Need 3 successes to close from half-open
    timeout=120,          # Wait 120 seconds before trying again
    expected_exceptions=(ConnectionError, TimeoutError)
)

@resilient_api_call("my_service", circuit_config=custom_config)
def my_service_call():
    pass
```

### Graceful Degradation

```python
from Agent.core.resilience import with_fallback, FallbackResult

@with_fallback(fallback_value=[], log_error=True)
def get_recommendations():
    # Try to get AI recommendations
    return openai_client.generate_completion([...])

result = get_recommendations()

if result.is_fallback:
    logger.warning("using_fallback", error=str(result.error))
    # Handle fallback case
else:
    # Use normal result
    recommendations = result.value
```

### Error Context Manager

```python
from Agent.core.resilience import ErrorContext

with ErrorContext("data_processing", raise_on_error=False, log_level="warning"):
    # Operations that might fail
    process_data()
    transform_data()
    
# Even if operations fail, execution continues (raise_on_error=False)
```

### Logging Decorators

```python
from Agent.config.logging_config import log_function_call, log_api_call, get_logger

logger = get_logger(__name__)

# Log function entry/exit with timing
@log_function_call(logger)
def complex_calculation(x, y):
    return x + y

# Log API calls with timing
@log_api_call(logger, "external_api")
def call_external_api():
    pass
```

---

## Configuration

### Environment Variables

Add to your `.env` file:

```env
# Logging
LOG_LEVEL=INFO
JSON_LOGS=true
LOG_FILE=logs/application.log
ENVIRONMENT=production

# Resilience
MAX_RETRIES=3
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60
```

### Settings Integration

The resilience features are automatically configured through the existing settings:

```python
from Agent.config import get_settings

settings = get_settings()

# These settings control retry behavior
settings.max_retries = 3  # Add to Settings class if needed
```

---

## Monitoring & Observability

### Circuit Breaker Status

```python
from Agent.core.resilience import circuit_registry

# Get all circuit breaker states
states = circuit_registry.get_all_states()

for name, state in states.items():
    print(f"{name}: {state['state']} (failures: {state['failure_count']})")

# Reset all circuit breakers (use with caution)
circuit_registry.reset_all()
```

### Structured Log Analysis

With JSON logging enabled, logs can be easily parsed and analyzed:

```json
{
  "event": "api_call_success",
  "api": "openai",
  "function": "generate_embedding",
  "duration_seconds": 0.823,
  "correlation_id": "corr-abc123def456",
  "service": "rules-engine-agent",
  "environment": "production",
  "timestamp": "2025-10-21T14:30:15.123456",
  "level": "info"
}
```

Use tools like:
- **Elasticsearch/Kibana**: For log aggregation and search
- **Datadog/New Relic**: For APM and monitoring
- **CloudWatch/Stackdriver**: For cloud-native logging

---

## Best Practices

### 1. Always Use Correlation IDs

```python
with CorrelationIdContext() as correlation_id:
    # All operations in this context share the same correlation_id
    file_analysis = analyze_file(file_path)
    rag_results = run_rag_pipeline(file_analysis)
    mappings = generate_mappings(rag_results)
```

### 2. Set Appropriate Timeouts

```python
# Different services may need different configurations
openai_config = CircuitBreakerConfig(
    failure_threshold=5,
    timeout=60  # 1 minute
)

database_config = CircuitBreakerConfig(
    failure_threshold=3,
    timeout=30  # 30 seconds - fail faster for local services
)
```

### 3. Log Structured Data

```python
# Good - structured
logger.info("file_processed", 
    file_name=file_name, 
    row_count=1000, 
    processing_time=2.5
)

# Bad - unstructured string
logger.info(f"Processed {file_name} with 1000 rows in 2.5s")
```

### 4. Handle Circuit Breaker Errors

```python
from Agent.core.resilience import CircuitBreakerError

try:
    result = openai_client.generate_embedding(text)
except CircuitBreakerError:
    # Service is down, use fallback or notify user
    logger.error("service_unavailable", service="openai")
    return use_fallback_strategy()
```

---

## Testing

### Test Retry Logic

```python
from unittest.mock import Mock, patch
from Agent.core.resilience import retry_on_api_error

@retry_on_api_error(max_attempts=3)
def flaky_function():
    # Will succeed on 3rd attempt
    pass

# Test that it retries correctly
with patch('some_module.api_call') as mock_call:
    mock_call.side_effect = [ConnectionError(), ConnectionError(), {"success": True}]
    result = flaky_function()
    assert mock_call.call_count == 3
```

### Test Circuit Breaker

```python
from Agent.core.resilience import CircuitBreaker, CircuitBreakerConfig, CircuitState

def test_circuit_breaker():
    breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))
    
    # Simulate failures
    for _ in range(2):
        try:
            breaker.call(lambda: 1/0)
        except:
            pass
    
    # Circuit should be open
    assert breaker.state == CircuitState.OPEN
```

---

## Migration Guide

### Updating Existing Code

**Before:**
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
vector = embeddings.embed_query(text)
```

**After:**
```python
from Agent.services.clients import ResilientOpenAIClient

client = ResilientOpenAIClient(
    api_key=settings.openai_api_key,
    max_retries=3
)
vector = client.generate_embedding(text)  # Now has retry + circuit breaker
```

---

## Troubleshooting

### Circuit Breaker Stuck Open

If a circuit breaker remains open:

1. Check service health: `health_service.check_openai(client)`
2. Review error logs for root cause
3. Manually reset if needed: `circuit_registry.get_or_create("openai").reset()`

### Excessive Retries

If you see too many retry attempts:

1. Reduce `max_attempts` parameter
2. Increase circuit breaker `failure_threshold` 
3. Check if errors are actually recoverable

### JSON Logs Not Appearing

Ensure you called `setup_structured_logging()` at application startup:

```python
# In main.py or __init__.py
from Agent.config.logging_config import setup_structured_logging

setup_structured_logging(json_logs=True)
```

---

## Performance Impact

- **Retry Logic**: Adds latency only on failures (exponential backoff)
- **Circuit Breaker**: Minimal overhead (~0.1ms per call)
- **Structured Logging**: ~5% overhead compared to basic logging
- **Health Checks**: Run on-demand, no continuous overhead

---

## Future Enhancements

Potential improvements for the future:

1. **Distributed Tracing**: OpenTelemetry integration
2. **Metrics Collection**: Prometheus metrics for API calls
3. **Rate Limiting**: Token bucket algorithm for API rate limits
4. **Connection Pooling**: Reuse HTTP connections
5. **Async/Await**: Non-blocking I/O for better performance
6. **Caching**: Redis-based caching for embeddings and results

---

For questions or issues, please refer to the main project documentation.
