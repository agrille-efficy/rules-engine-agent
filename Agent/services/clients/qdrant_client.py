"""
Resilient Qdrant client wrapper with retry logic and circuit breaker.
"""
from typing import Optional, List, Any, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, SearchRequest

from ...config.logging_config import get_logger, log_api_call
from ...core.resilience import (
    resilient_api_call,
    CircuitBreakerConfig,
    QdrantServiceError,
    ErrorContext
)

logger = get_logger(__name__)


class ResilientQdrantClient:
    """
    Wrapper for Qdrant API calls with built-in resilience patterns.
    
    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker to prevent cascading failures
    - Structured logging for all API calls
    - Error handling and classification
    """
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        circuit_config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize resilient Qdrant client.
        
        Args:
            url: Qdrant server URL
            api_key: Optional API key
            max_retries: Maximum retry attempts (default: 3)
            circuit_config: Optional circuit breaker configuration
        """
        self.url = url
        self.api_key = api_key
        self.max_retries = max_retries
        
        # Configure circuit breaker for Qdrant
        self.circuit_config = circuit_config or CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=60,
            expected_exceptions=(Exception,)
        )
        
        # Initialize client
        self._client: Optional[QdrantClient] = None
        
        logger.info(
            "qdrant_client_initialized",
            url=url,
            max_retries=max_retries
        )
    
    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key
            )
        return self._client
    
    @resilient_api_call("qdrant", max_attempts=3)
    @log_api_call(logger, "qdrant")
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        **kwargs
    ) -> List[Any]:
        """
        Search vector store with resilience.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            query_filter: Optional filter
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
            
        Raises:
            QdrantServiceError: If API call fails after retries
        """
        try:
            with ErrorContext("qdrant_search", raise_on_error=True):
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=query_filter,
                    **kwargs
                )
                
                logger.debug(
                    "qdrant_search_success",
                    collection=collection_name,
                    result_count=len(results),
                    limit=limit
                )
                
                return results
                
        except Exception as e:
            logger.error(
                "qdrant_search_failed",
                collection=collection_name,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise QdrantServiceError(f"Qdrant search failed: {e}") from e
    
    @resilient_api_call("qdrant_upsert", max_attempts=3)
    @log_api_call(logger, "qdrant_upsert")
    def upsert(
        self,
        collection_name: str,
        points: List[PointStruct],
        **kwargs
    ) -> Any:
        """
        Upsert points to collection with resilience.
        
        Args:
            collection_name: Name of the collection
            points: List of points to upsert
            **kwargs: Additional upsert parameters
            
        Returns:
            Upsert operation result
            
        Raises:
            QdrantServiceError: If API call fails after retries
        """
        try:
            with ErrorContext("qdrant_upsert", raise_on_error=True):
                result = self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    **kwargs
                )
                
                logger.info(
                    "qdrant_upsert_success",
                    collection=collection_name,
                    point_count=len(points)
                )
                
                return result
                
        except Exception as e:
            logger.error(
                "qdrant_upsert_failed",
                collection=collection_name,
                point_count=len(points),
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise QdrantServiceError(f"Qdrant upsert failed: {e}") from e
    
    @resilient_api_call("qdrant_get_collection", max_attempts=3)
    @log_api_call(logger, "qdrant_get_collection")
    def get_collection(self, collection_name: str) -> Any:
        """
        Get collection information with resilience.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection information
            
        Raises:
            QdrantServiceError: If API call fails after retries
        """
        try:
            with ErrorContext("qdrant_get_collection", raise_on_error=True):
                collection_info = self.client.get_collection(collection_name)
                
                logger.debug(
                    "qdrant_get_collection_success",
                    collection=collection_name,
                    point_count=collection_info.points_count if hasattr(collection_info, 'points_count') else 0
                )
                
                return collection_info
                
        except Exception as e:
            logger.error(
                "qdrant_get_collection_failed",
                collection=collection_name,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise QdrantServiceError(f"Qdrant get collection failed: {e}") from e
    
    @resilient_api_call("qdrant_get_collections", max_attempts=3)
    @log_api_call(logger, "qdrant_get_collections")
    def get_collections(self) -> Any:
        """
        Get all collections with resilience.
        
        Returns:
            Collections information
            
        Raises:
            QdrantServiceError: If API call fails after retries
        """
        try:
            with ErrorContext("qdrant_get_collections", raise_on_error=True):
                collections = self.client.get_collections()
                
                collection_names = [col.name for col in collections.collections] if hasattr(collections, 'collections') else []
                
                logger.debug(
                    "qdrant_get_collections_success",
                    collection_count=len(collection_names),
                    collections=collection_names
                )
                
                return collections
                
        except Exception as e:
            logger.error(
                "qdrant_get_collections_failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise QdrantServiceError(f"Qdrant get collections failed: {e}") from e
    
    @resilient_api_call("qdrant_create_collection", max_attempts=3)
    @log_api_call(logger, "qdrant_create_collection")
    def create_collection(self, collection_name: str, **kwargs) -> Any:
        """
        Create collection with resilience.
        
        Args:
            collection_name: Name of the collection
            **kwargs: Collection configuration parameters
            
        Returns:
            Create operation result
            
        Raises:
            QdrantServiceError: If API call fails after retries
        """
        try:
            with ErrorContext("qdrant_create_collection", raise_on_error=True):
                result = self.client.create_collection(
                    collection_name=collection_name,
                    **kwargs
                )
                
                logger.info(
                    "qdrant_create_collection_success",
                    collection=collection_name
                )
                
                return result
                
        except Exception as e:
            logger.error(
                "qdrant_create_collection_failed",
                collection=collection_name,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise QdrantServiceError(f"Qdrant create collection failed: {e}") from e
    
    @resilient_api_call("qdrant_create_payload_index", max_attempts=3)
    @log_api_call(logger, "qdrant_create_payload_index")
    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Any,
        **kwargs
    ) -> Any:
        """
        Create payload index with resilience.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field to index
            field_schema: Schema for the field
            **kwargs: Additional index parameters
            
        Returns:
            Create index operation result
            
        Raises:
            QdrantServiceError: If API call fails after retries
        """
        try:
            with ErrorContext("qdrant_create_payload_index", raise_on_error=True):
                result = self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    **kwargs
                )
                
                logger.info(
                    "qdrant_create_payload_index_success",
                    collection=collection_name,
                    field=field_name
                )
                
                return result
                
        except Exception as e:
            logger.error(
                "qdrant_create_payload_index_failed",
                collection=collection_name,
                field=field_name,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise QdrantServiceError(f"Qdrant create payload index failed: {e}") from e
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client configuration information."""
        return {
            "url": self.url,
            "max_retries": self.max_retries,
            "circuit_config": {
                "failure_threshold": self.circuit_config.failure_threshold,
                "success_threshold": self.circuit_config.success_threshold,
                "timeout": self.circuit_config.timeout
            }
        }
