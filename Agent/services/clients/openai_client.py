"""
Resilient OpenAI client wrapper with retry logic and circuit breaker.
"""
from typing import Optional, List, Any, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ...config.logging_config import get_logger, log_api_call
from ...core.resilience import (
    resilient_api_call,
    CircuitBreakerConfig,
    OpenAIServiceError,
    ErrorContext
)

logger = get_logger(__name__)


class ResilientOpenAIClient:
    """
    Wrapper for OpenAI API calls with built-in resilience patterns.
    
    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker to prevent cascading failures
    - Structured logging for all API calls
    - Error handling and classification
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.3,
        max_retries: int = 3,
        circuit_config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize resilient OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4)
            temperature: Temperature for generation (default: 0.3)
            max_retries: Maximum retry attempts (default: 3)
            circuit_config: Optional circuit breaker configuration
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Configure circuit breaker for OpenAI
        self.circuit_config = circuit_config or CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=60,
            expected_exceptions=(Exception,)
        )
        
        # Initialize clients
        self._chat_client: Optional[ChatOpenAI] = None
        self._embeddings_client: Optional[OpenAIEmbeddings] = None
        
        logger.info(
            "openai_client_initialized",
            model=model,
            temperature=temperature,
            max_retries=max_retries
        )
    
    @property
    def chat(self) -> ChatOpenAI:
        """Get or create chat client."""
        if self._chat_client is None:
            self._chat_client = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                max_retries=self.max_retries
            )
        return self._chat_client
    
    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Get or create embeddings client."""
        if self._embeddings_client is None:
            self._embeddings_client = OpenAIEmbeddings(
                api_key=self.api_key
            )
        return self._embeddings_client
    
    @resilient_api_call("openai", max_attempts=3)
    @log_api_call(logger, "openai")
    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate chat completion with resilience.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional arguments for ChatOpenAI
            
        Returns:
            Generated text response
            
        Raises:
            OpenAIServiceError: If API call fails after retries
        """
        try:
            with ErrorContext("openai_completion", raise_on_error=True):
                from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
                
                # Convert dict messages to LangChain message objects
                lc_messages = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        lc_messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        lc_messages.append(AIMessage(content=content))
                    else:
                        lc_messages.append(HumanMessage(content=content))
                
                response = self.chat.invoke(lc_messages, **kwargs)
                
                logger.debug(
                    "openai_completion_success",
                    message_count=len(messages),
                    response_length=len(response.content)
                )
                
                return response.content
                
        except Exception as e:
            logger.error(
                "openai_completion_failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise OpenAIServiceError(f"OpenAI completion failed: {e}") from e
    
    @resilient_api_call("openai_embeddings", max_attempts=3)
    @log_api_call(logger, "openai_embeddings")
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text with resilience.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            OpenAIServiceError: If API call fails after retries
        """
        try:
            with ErrorContext("openai_embedding", raise_on_error=True):
                embedding = self.embeddings.embed_query(text)
                
                logger.debug(
                    "openai_embedding_success",
                    text_length=len(text),
                    embedding_dimension=len(embedding)
                )
                
                return embedding
                
        except Exception as e:
            logger.error(
                "openai_embedding_failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise OpenAIServiceError(f"OpenAI embedding failed: {e}") from e
    
    @resilient_api_call("openai_embeddings_batch", max_attempts=3)
    @log_api_call(logger, "openai_embeddings_batch")
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with resilience.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            OpenAIServiceError: If API call fails after retries
        """
        try:
            with ErrorContext("openai_embeddings_batch", raise_on_error=True):
                embeddings = self.embeddings.embed_documents(texts)
                
                logger.debug(
                    "openai_embeddings_batch_success",
                    text_count=len(texts),
                    embedding_dimension=len(embeddings[0]) if embeddings else 0
                )
                
                return embeddings
                
        except Exception as e:
            logger.error(
                "openai_embeddings_batch_failed",
                text_count=len(texts),
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise OpenAIServiceError(f"OpenAI batch embeddings failed: {e}") from e
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client configuration information."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "circuit_config": {
                "failure_threshold": self.circuit_config.failure_threshold,
                "success_threshold": self.circuit_config.success_threshold,
                "timeout": self.circuit_config.timeout
            }
        }
