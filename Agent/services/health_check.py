"""
Health check and monitoring utilities for external services.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from ..config.logging_config import get_logger
from ..core.resilience import circuit_registry

logger = get_logger(__name__)


class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheck:
    """Health check result for a service."""
    
    def __init__(
        self,
        service_name: str,
        status: ServiceStatus,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        response_time_ms: Optional[float] = None
    ):
        self.service_name = service_name
        self.status = status
        self.message = message
        self.details = details or {}
        self.response_time_ms = response_time_ms
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health check to dictionary."""
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp.isoformat()
        }
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status == ServiceStatus.HEALTHY


class HealthCheckService:
    """Service for checking health of external dependencies."""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
    
    def check_openai(self, client) -> HealthCheck:
        """
        Check OpenAI service health.
        
        Args:
            client: ResilientOpenAIClient instance
            
        Returns:
            HealthCheck result
        """
        import time
        
        try:
            start_time = time.time()
            
            # Try a simple embedding call
            result = client.generate_embedding("health check")
            
            response_time = (time.time() - start_time) * 1000
            
            if response_time > 5000:  # > 5 seconds
                status = ServiceStatus.DEGRADED
                message = "OpenAI responding slowly"
            else:
                status = ServiceStatus.HEALTHY
                message = "OpenAI service operational"
            
            check = HealthCheck(
                service_name="openai",
                status=status,
                message=message,
                response_time_ms=round(response_time, 2),
                details={
                    "embedding_dimension": len(result) if result else 0,
                    "model": client.model
                }
            )
            
            logger.info(
                "health_check_completed",
                service="openai",
                status=status.value,
                response_time_ms=round(response_time, 2)
            )
            
            self.checks["openai"] = check
            return check
            
        except Exception as e:
            check = HealthCheck(
                service_name="openai",
                status=ServiceStatus.UNHEALTHY,
                message=f"OpenAI service error: {str(e)}",
                details={"error_type": type(e).__name__}
            )
            
            logger.error(
                "health_check_failed",
                service="openai",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            self.checks["openai"] = check
            return check
    
    def check_qdrant(self, client) -> HealthCheck:
        """
        Check Qdrant service health.
        
        Args:
            client: ResilientQdrantClient instance
            
        Returns:
            HealthCheck result
        """
        import time
        
        try:
            start_time = time.time()
            
            # Try to list collections
            collections = client.get_collections()
            
            response_time = (time.time() - start_time) * 1000
            
            collection_names = [col.name for col in collections.collections] if hasattr(collections, 'collections') else []
            
            if response_time > 3000:  # > 3 seconds
                status = ServiceStatus.DEGRADED
                message = "Qdrant responding slowly"
            else:
                status = ServiceStatus.HEALTHY
                message = "Qdrant service operational"
            
            check = HealthCheck(
                service_name="qdrant",
                status=status,
                message=message,
                response_time_ms=round(response_time, 2),
                details={
                    "collection_count": len(collection_names),
                    "collections": collection_names[:5]  # First 5
                }
            )
            
            logger.info(
                "health_check_completed",
                service="qdrant",
                status=status.value,
                response_time_ms=round(response_time, 2)
            )
            
            self.checks["qdrant"] = check
            return check
            
        except Exception as e:
            check = HealthCheck(
                service_name="qdrant",
                status=ServiceStatus.UNHEALTHY,
                message=f"Qdrant service error: {str(e)}",
                details={"error_type": type(e).__name__}
            )
            
            logger.error(
                "health_check_failed",
                service="qdrant",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            self.checks["qdrant"] = check
            return check
    
    def check_all_circuit_breakers(self) -> Dict[str, Any]:
        """
        Check status of all circuit breakers.
        
        Returns:
            Dictionary of circuit breaker states
        """
        states = circuit_registry.get_all_states()
        
        logger.info(
            "circuit_breaker_status_check",
            breaker_count=len(states),
            states=states
        )
        
        return states
    
    def get_overall_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dictionary with overall health information
        """
        if not self.checks:
            return {
                "status": ServiceStatus.UNKNOWN.value,
                "message": "No health checks have been performed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Determine overall status
        all_healthy = all(check.is_healthy() for check in self.checks.values())
        any_unhealthy = any(check.status == ServiceStatus.UNHEALTHY for check in self.checks.values())
        
        if any_unhealthy:
            overall_status = ServiceStatus.UNHEALTHY
            message = "One or more services are unhealthy"
        elif all_healthy:
            overall_status = ServiceStatus.HEALTHY
            message = "All services are operational"
        else:
            overall_status = ServiceStatus.DEGRADED
            message = "Some services are degraded"
        
        circuit_breakers = self.check_all_circuit_breakers()
        
        return {
            "status": overall_status.value,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "services": {
                name: check.to_dict()
                for name, check in self.checks.items()
            },
            "circuit_breakers": circuit_breakers
        }
    
    def print_health_report(self):
        """Print formatted health report to console."""
        overall = self.get_overall_health()
        
        print("\n" + "=" * 80)
        print("SYSTEM HEALTH REPORT")
        print("=" * 80)
        print(f"Overall Status: {overall['status'].upper()}")
        print(f"Timestamp: {overall['timestamp']}")
        print(f"Message: {overall['message']}")
        
        if overall.get('services'):
            print("\n" + "-" * 80)
            print("SERVICE HEALTH")
            print("-" * 80)
            
            for name, check in overall['services'].items():
                status_icon = {
                    "healthy": "✅",
                    "degraded": "⚠️",
                    "unhealthy": "❌",
                    "unknown": "❓"
                }.get(check['status'], "❓")
                
                print(f"\n{status_icon} {name.upper()}")
                print(f"   Status: {check['status']}")
                print(f"   Message: {check['message']}")
                if check.get('response_time_ms'):
                    print(f"   Response Time: {check['response_time_ms']}ms")
                if check.get('details'):
                    print(f"   Details: {check['details']}")
        
        if overall.get('circuit_breakers'):
            print("\n" + "-" * 80)
            print("CIRCUIT BREAKERS")
            print("-" * 80)
            
            for name, state in overall['circuit_breakers'].items():
                state_icon = {
                    "closed": "✅",
                    "half_open": "⚠️",
                    "open": "❌"
                }.get(state['state'], "❓")
                
                print(f"\n{state_icon} {name.upper()}")
                print(f"   State: {state['state']}")
                print(f"   Failures: {state['failure_count']}")
                print(f"   Last Change: {state['last_state_change']}")
        
        print("\n" + "=" * 80 + "\n")


# Global health check service instance
health_service = HealthCheckService()
