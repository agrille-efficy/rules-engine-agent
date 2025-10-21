"""
Example script demonstrating resilience and structured logging features.

This script shows how to:
1. Initialize structured logging with correlation IDs
2. Use resilient API clients with retry and circuit breakers
3. Perform health checks on services
4. Handle errors gracefully with fallbacks
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Agent.config import get_settings
from Agent.config.logging_config import (
    setup_structured_logging,
    get_logger,
    CorrelationIdContext
)
from Agent.services.clients import ResilientOpenAIClient, ResilientQdrantClient
from Agent.services.health_check import health_service
from Agent.core.resilience import (
    CircuitBreakerError,
    OpenAIServiceError,
    QdrantServiceError,
    circuit_registry
)


def main():
    """Main demonstration function."""
    
    # ========================================================================
    # 1. SETUP STRUCTURED LOGGING
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Setting up structured logging")
    print("=" * 80)
    
    setup_structured_logging(
        log_level="INFO",
        json_logs=False,  # Use human-readable format for demo
        service_name="resilience-demo"
    )
    
    logger = get_logger(__name__)
    logger.info("demo_started", version="1.0")
    
    # ========================================================================
    # 2. INITIALIZE RESILIENT CLIENTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Initializing resilient API clients")
    print("=" * 80)
    
    settings = get_settings()
    
    # Create resilient OpenAI client
    openai_client = ResilientOpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=settings.temperature,
        max_retries=3
    )
    logger.info("openai_client_created", model=settings.llm_model)
    
    # Create resilient Qdrant client
    qdrant_client = ResilientQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        max_retries=3
    )
    logger.info("qdrant_client_created", url=settings.qdrant_url)
    
    # ========================================================================
    # 3. HEALTH CHECKS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Running health checks")
    print("=" * 80)
    
    with CorrelationIdContext() as correlation_id:
        logger.info("health_check_started", correlation_id=correlation_id)
        
        # Check OpenAI
        openai_health = health_service.check_openai(openai_client)
        logger.info(
            "health_check_result",
            service="openai",
            status=openai_health.status.value,
            response_time_ms=openai_health.response_time_ms
        )
        
        # Check Qdrant
        qdrant_health = health_service.check_qdrant(qdrant_client)
        logger.info(
            "health_check_result",
            service="qdrant",
            status=qdrant_health.status.value,
            response_time_ms=qdrant_health.response_time_ms
        )
        
        # Print health report
        health_service.print_health_report()
    
    # ========================================================================
    # 4. DEMONSTRATE RESILIENT API CALLS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Demonstrating resilient API calls")
    print("=" * 80)
    
    with CorrelationIdContext() as correlation_id:
        logger.info("api_demo_started", correlation_id=correlation_id)
        
        try:
            # Example 1: Generate embedding with automatic retry
            logger.info("generating_embedding", text="sample text for embedding")
            embedding = openai_client.generate_embedding("sample text for embedding")
            logger.info("embedding_generated", dimension=len(embedding))
            print(f"✅ Generated embedding with dimension: {len(embedding)}")
            
            # Example 2: Generate batch embeddings
            texts = ["first text", "second text", "third text"]
            logger.info("generating_batch_embeddings", count=len(texts))
            embeddings = openai_client.generate_embeddings_batch(texts)
            logger.info("batch_embeddings_generated", count=len(embeddings))
            print(f"✅ Generated {len(embeddings)} embeddings")
            
            # Example 3: Search Qdrant
            logger.info("searching_qdrant", collection=settings.qdrant_collection_name)
            results = qdrant_client.search(
                collection_name=settings.qdrant_collection_name,
                query_vector=embedding,
                limit=5
            )
            logger.info("qdrant_search_completed", result_count=len(results))
            print(f"✅ Found {len(results)} results from Qdrant")
            
        except CircuitBreakerError as e:
            logger.error("circuit_breaker_open", error=str(e))
            print(f"❌ Circuit breaker is open: {e}")
            
        except OpenAIServiceError as e:
            logger.error("openai_service_error", error=str(e))
            print(f"❌ OpenAI service failed after retries: {e}")
            
        except QdrantServiceError as e:
            logger.error("qdrant_service_error", error=str(e))
            print(f"❌ Qdrant service failed after retries: {e}")
    
    # ========================================================================
    # 5. DEMONSTRATE CHAT COMPLETION
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Demonstrating chat completion with retry")
    print("=" * 80)
    
    with CorrelationIdContext() as correlation_id:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain circuit breaker pattern in one sentence."}
            ]
            
            logger.info("generating_completion", message_count=len(messages))
            response = openai_client.generate_completion(messages)
            logger.info("completion_generated", response_length=len(response))
            
            print(f"\n💬 AI Response:")
            print(f"   {response}")
            
        except Exception as e:
            logger.error("completion_failed", error=str(e))
            print(f"❌ Chat completion failed: {e}")
    
    # ========================================================================
    # 6. CIRCUIT BREAKER STATUS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Circuit breaker status")
    print("=" * 80)
    
    circuit_states = circuit_registry.get_all_states()
    
    for name, state in circuit_states.items():
        status_icon = {
            "closed": "✅",
            "half_open": "⚠️",
            "open": "❌"
        }.get(state['state'], "❓")
        
        print(f"{status_icon} {name.upper()}")
        print(f"   State: {state['state']}")
        print(f"   Failures: {state['failure_count']}")
        print(f"   Last Change: {state['last_state_change']}")
        print()
    
    # ========================================================================
    # 7. CLIENT INFORMATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Client configuration")
    print("=" * 80)
    
    openai_info = openai_client.get_client_info()
    print("\n📊 OpenAI Client:")
    print(f"   Model: {openai_info['model']}")
    print(f"   Temperature: {openai_info['temperature']}")
    print(f"   Max Retries: {openai_info['max_retries']}")
    print(f"   Circuit Breaker Threshold: {openai_info['circuit_config']['failure_threshold']}")
    
    qdrant_info = qdrant_client.get_client_info()
    print("\n📊 Qdrant Client:")
    print(f"   URL: {qdrant_info['url']}")
    print(f"   Max Retries: {qdrant_info['max_retries']}")
    print(f"   Circuit Breaker Threshold: {qdrant_info['circuit_config']['failure_threshold']}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\n✅ All resilience features demonstrated:")
    print("   • Structured logging with correlation IDs")
    print("   • Resilient API clients with retry logic")
    print("   • Circuit breaker pattern implementation")
    print("   • Health checks and monitoring")
    print("   • Error handling and graceful degradation")
    print("\n📖 See RESILIENCE_GUIDE.md for detailed documentation")
    print("=" * 80 + "\n")
    
    logger.info("demo_completed", status="success")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
