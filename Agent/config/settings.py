import os
from typing import Optional 
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from .constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_REFINEMENTS,
    DEFAULT_RAG_COLLECTION
)

class Settings(BaseModel):
    """
    Application settings with Pydantic validation.
    Ensures type safety and validates configuration at startup.
    """
    
    # API Keys (required)
    openai_api_key: str = Field(..., min_length=20)
    qdrant_url: str = Field(..., min_length=1)
    qdrant_api_key: Optional[str] = None
    
    # LLM Configuration
    llm_model: str = Field(default=DEFAULT_LLM_MODEL)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    
    # RAG Configuration
    qdrant_collection_name: str = Field(default=DEFAULT_RAG_COLLECTION)
    top_k_results: int = Field(default=10, ge=1, le=100)
    max_refinements: int = Field(default=DEFAULT_MAX_REFINEMENTS, ge=1, le=10)
    
    # Efficy Configuration (optional)
    efficy_customer: Optional[str] = None
    efficy_base_url: Optional[str] = None
    efficy_user: Optional[str] = None
    efficy_password: Optional[str] = None
    
    model_config = {
        'validate_assignment': True,  # Validate on attribute assignment
        'arbitrary_types_allowed': False
    }

    @field_validator('openai_api_key')
    @classmethod
    def validate_openai_key(cls, v):
        """Validate OpenAI API key format"""
        if not v.startswith('sk-'):
            raise ValueError("Invalid OpenAI API key format (must start with 'sk-')")
        return v
    
    @field_validator('llm_model')
    @classmethod
    def validate_model_name(cls, v):
        """Validate LLM model name"""
        allowed_prefixes = ['gpt-4o', 'gpt-4o']
        if not any(v.startswith(prefix) for prefix in allowed_prefixes):
            raise ValueError(f"Unsupported model: {v}. Must be GPT-3.5 or GPT-4o variant")
        return v
    
    @field_validator('qdrant_url')
    @classmethod
    def validate_qdrant_url(cls, v):
        """Validate Qdrant URL format"""
        if not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError("Qdrant URL must start with http:// or https://")
        return v
    
    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> 'Settings':
        """
        Load settings from environment variables.
        
        Args:
            env_file: Optional path to .env file
            
        Returns:
            Validated Settings instance
            
        Raises:
            ValueError: If required settings are missing or invalid
        """
        if env_file and env_file.exists():
            load_dotenv(dotenv_path=env_file)
        else: 
            default_env = Path(__file__).parent.parent / ".env"
            if default_env.exists():
                load_dotenv(dotenv_path=default_env)
        
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            qdrant_url=os.getenv("QDRANT_URL", ""),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            temperature=float(os.getenv("TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", DEFAULT_RAG_COLLECTION),
            top_k_results=int(os.getenv("TOP_K_RESULTS", "10")),
            max_refinements=int(os.getenv("MAX_REFINEMENTS", str(DEFAULT_MAX_REFINEMENTS))),
            efficy_customer=os.getenv("EFFICY_CUSTOMER"),
            efficy_base_url=os.getenv("EFFICY_BASE_URL"),
            efficy_user=os.getenv("EFFICY_USER"),
            efficy_password=os.getenv("EFFICY_PASSWORD"),
        )

_settings_instance: Optional[Settings] = None 

def get_settings(env_file: Optional[Path] = None) -> Settings:
    """
    Get settings instance (singleton pattern).
    
    Args:
        env_file: Optional path to .env file
        
    Returns:
        Validated Settings instance
    """
    global _settings_instance 
    if _settings_instance is None:
        _settings_instance = Settings.from_env(env_file=env_file) 
    return _settings_instance
