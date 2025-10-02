import os
from typing import Optional 
from pathlib import Path
from dotenv import load_dotenv

from .constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_REFINEMENTS,
    DEFAULT_RAG_COLLECTION
)

class Settings:
    """
    Application settings loaded from environment variables.
    Uses defaults from constants.py when env vars not set.
    """

    def __init__(self, env_file: Optional[Path] = None):
        """
        Load settings from environment.

        Args:
            env_file: Optional path to .env file.
        """
        if env_file and env_file.exists():
            load_dotenv(dotenv_path=env_file)
        else: 
            default_env = Path(__file__).parent.parent / ".env"
            if default_env.exists():
                load_dotenv(dotenv_path=default_env)
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        self.llm_model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
        self.temperature = float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE))

        self.rag_collection_name = os.getenv("RAG_COLLECTION_NAME", DEFAULT_RAG_COLLECTION)
        self.top_k_results = int(os.getenv("TOP_K_RESULTS", "10"))

        self.max_refinements = int(os.getenv("MAX_REFINEMENTS", str(DEFAULT_MAX_REFINEMENTS)))

        self._validate()

    def _validate(self):
        """validate that required settings are present."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file")
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL is required. Please set it in your .env file")
        
_settings_instance: Optional[Settings] = None 

def get_settings(env_file: Optional[Path] = None) -> Settings:
    """
    Get settings instance
    
    Args:
        env_file: Optional path to .env file.
        
    Returns:
        Settings instance
    """

    global _settings_instance 
    if _settings_instance is None:
        _settings_instance = Settings(env_file=env_file) 
    return _settings_instance
