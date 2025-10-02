import os
from dotenv import load_dotenv
from typing import Optional

class Config:
    """Secure configuration management"""
    
    def __init__(self):
        load_dotenv()
        self._validate_required_env_vars()
    
    def _validate_required_env_vars(self):
        """Validate all required environment variables are present"""
        # Only require OpenAI and Qdrant vars - EFFICY vars are optional with defaults
        required_vars = [
            'OPENAI_API_KEY',
            'QDRANT_URL', 
            'QDRANT_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    @property
    def openai_api_key(self) -> str:
        return os.getenv('OPENAI_API_KEY')
    
    @property
    def qdrant_url(self) -> str:
        return os.getenv('QDRANT_URL')
    
    @property
    def qdrant_api_key(self) -> str:
        return os.getenv('QDRANT_API_KEY')
    
    @property
    def efficy_base_url(self) -> str:
        return os.getenv('EFFICY_BASE_URL', 'https://sandbox-5.efficytest.cloud')
    
    @property
    def efficy_customer(self) -> str:
        return os.getenv('EFFICY_CUSTOMER', 'SANDBOX05')
    
    @property
    def efficy_username(self) -> str:
        return os.getenv('EFFICY_USERNAME', 'paul')
    
    @property
    def efficy_password(self) -> str:
        return os.getenv('EFFICY_PASSWORD', 'Eff1cyDemo!')

# Global config instance
config = Config()