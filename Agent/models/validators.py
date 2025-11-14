"""
Input validation and sanitization using Pydantic.
Protects against prompt injection, path traversal, and other security vulnerabilities.
"""
import re
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class FileTypeEnum(str, Enum):
    """Allowed file types for processing"""
    CSV = "csv"
    EXCEL = "excel"
    XLSX = "xlsx"
    XLS = "xls"
    JSON = "json"
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    XML = "xml"


class UserContextInput(BaseModel):
    """
    Validated user context input with sanitization.
    Prevents prompt injection attacks.
    """
    raw_context: str = Field(..., min_length=1, max_length=2000)
    
    @field_validator('raw_context')
    @classmethod
    def sanitize_context(cls, v):
        """Sanitize user context to prevent prompt injection"""
        if not v or not v.strip():
            raise ValueError("Context cannot be empty")
        
        # Remove null bytes
        v = v.replace('\x00', '')
        
        # Detect and block potential prompt injection patterns
        injection_patterns = [
            r'ignore\s+(all|previous|prior|above)\s+(previous\s+)?(instructions|prompts|commands)',
            r'system\s*:',
            r'assistant\s*:',
            r'<\|.*?\|>',  # Special tokens
            r'\[INST\]',   # Instruction markers
            r'\[/INST\]',
            r'<s>',        # Special model tokens
            r'</s>',
            r'###',        # Common instruction separators
            r'---\s*\n',   # Markdown separators used in prompts
            r'Act\s+as\s+a\s+',  # Role-playing prompts
            r'You\s+are\s+now\s+',
            r'Forget\s+(everything|all|previous)',
            r'jailbreak',
            r'DAN\s+mode',
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(
                    f"Input contains potentially malicious pattern. "
                    f"Please provide a simple description without special instructions."
                )
        
        # Limit special characters
        special_char_count = sum(1 for c in v if not c.isalnum() and not c.isspace() and c not in '.,;:!?-_()[]')
        if special_char_count > len(v) * 0.3:
            raise ValueError("Input contains excessive special characters")
        
        # Strip leading/trailing whitespace
        v = v.strip()
        
        # Limit to reasonable length per line
        lines = v.split('\n')
        if len(lines) > 20:
            raise ValueError("Input has too many lines (max 20)")
        
        for line in lines:
            if len(line) > 200:
                raise ValueError("Individual line too long (max 200 characters)")
        
        return v
    
    def get_sanitized(self) -> str:
        """Get the sanitized context string"""
        return self.raw_context


class FilePathInput(BaseModel):
    """
    Validated file path input with path traversal protection.
    """
    file_path: str = Field(..., min_length=1, max_length=500)
    workspace_root: Optional[str] = None
    
    @field_validator('file_path')
    @classmethod
    def validate_path(cls, v):
        """Validate and sanitize file path"""
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        
        # Remove null bytes
        v = v.replace('\x00', '')
        
        # Detect path traversal attempts
        dangerous_patterns = [
            '..',
            '~',
            '$',
            '${',
            '`',
            '|',
            ';',
            '&',
            '\n',
            '\r',
        ]
        
        for pattern in dangerous_patterns:
            if pattern in v:
                raise ValueError(f"File path contains dangerous pattern: {pattern}")
        
        # Check for absolute path indicators that might escape workspace
        if v.startswith('\\\\') or v.startswith('//'):
            raise ValueError("UNC paths are not allowed")
        
        return v.strip()
    
    @model_validator(mode='after')
    def validate_within_workspace(self):
        """Ensure file path is within workspace (if workspace_root provided)"""
        file_path = self.file_path
        workspace_root = self.workspace_root
        
        if not file_path:
            return self
        
        # Resolve to absolute path
        try:
            abs_path = Path(file_path).resolve()
            
            # Check file exists
            if not abs_path.exists():
                raise ValueError(f"File does not exist: {file_path}")
            
            # Check if it's a file (not a directory)
            if not abs_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            
            # Validate within workspace if provided
            if workspace_root:
                workspace_abs = Path(workspace_root).resolve()
                try:
                    abs_path.relative_to(workspace_abs)
                except ValueError:
                    raise ValueError(
                        f"File path is outside workspace. "
                        f"File: {abs_path}, Workspace: {workspace_abs}"
                    )
            
            # Update with resolved path
            self.file_path = str(abs_path)
            
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid file path: {str(e)}")
        
        return self
    
    def get_safe_path(self) -> Path:
        """Get validated Path object"""
        return Path(self.file_path)


class FileSizeValidator(BaseModel):
    """Validate file size constraints"""
    file_path: str
    max_size_mb: int = Field(default=100, ge=1, le=1000)
    size_bytes: Optional[int] = None
    size_mb: Optional[float] = None
    
    @model_validator(mode='after')
    def check_file_size(self):
        """Verify file size is within limits"""
        file_path = self.file_path
        max_size_mb = self.max_size_mb
        
        if not file_path or not os.path.exists(file_path):
            return self
        
        try:
            size_bytes = os.path.getsize(file_path)
            size_mb = size_bytes / (1024 * 1024)
            
            if size_mb > max_size_mb:
                raise ValueError(
                    f"File too large: {size_mb:.2f}MB (max {max_size_mb}MB). "
                    f"Please process in smaller chunks or increase limit."
                )
            
            self.size_bytes = size_bytes
            self.size_mb = size_mb
            
        except OSError as e:
            raise ValueError(f"Cannot read file size: {str(e)}")
        
        return self


class QueryInput(BaseModel):
    """
    Validated query input for semantic search.
    Prevents injection into vector search queries.
    """
    query: str = Field(..., min_length=1, max_length=1000)
    
    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v):
        """Sanitize search query"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        
        # Remove null bytes and control characters
        v = ''.join(char for char in v if char.isprintable() or char.isspace())
        v = v.replace('\x00', '')
        
        # Check for SQL injection patterns (even though we're using vector search)
        sql_patterns = [
            r"('|(\\'))",  # SQL quotes
            r'(--|#|/\*|\*/)',  # SQL comments
            r'\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b',
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Query contains potentially malicious SQL pattern")
        
        return v.strip()


class TableNameInput(BaseModel):
    """Validated table name input"""
    table_name: str = Field(..., min_length=1, max_length=200)
    
    @field_validator('table_name')
    @classmethod
    def validate_table_name(cls, v):
        """Validate table name format"""
        if not v or not v.strip():
            raise ValueError("Table name cannot be empty")
        
        # Only allow alphanumeric, underscore, and basic separators
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError(
                "Table name can only contain letters, numbers, underscores, hyphens, and dots"
            )
        
        return v.strip()


class ColumnNamesInput(BaseModel):
    """Validated column names list"""
    columns: List[str] = Field(..., min_length=1, max_length=1000)
    
    @field_validator('columns')
    @classmethod
    def validate_columns(cls, v):
        """Validate each column name"""
        if not v:
            raise ValueError("Column list cannot be empty")
        
        sanitized = []
        seen = set()
        
        for col in v:
            if not isinstance(col, str):
                raise ValueError(f"Column name must be string, got {type(col)}")
            
            # Remove null bytes and control characters
            col = ''.join(char for char in col if char.isprintable() or char.isspace())
            col = col.strip()
            
            if not col:
                continue
            
            if len(col) > 200:
                raise ValueError(f"Column name too long: {col[:50]}... (max 200 chars)")
            
            # Check for duplicates
            col_lower = col.lower()
            if col_lower in seen:
                continue
            
            seen.add(col_lower)
            sanitized.append(col)
        
        if not sanitized:
            raise ValueError("No valid column names after sanitization")
        
        return sanitized


class PipelineInput(BaseModel):
    """Complete validated input for RAG pipeline"""
    file_path: str
    user_context: Optional[str] = None
    workspace_root: Optional[str] = None
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v):
        """Validate file path"""
        validated = FilePathInput(file_path=v)
        return validated.file_path
    
    @field_validator('user_context')
    @classmethod
    def validate_context(cls, v):
        """Validate user context if provided"""
        if v is None or not v.strip():
            return None
        
        validated = UserContextInput(raw_context=v)
        return validated.get_sanitized()
    
    @model_validator(mode='after')
    def validate_file_size(self):
        """Validate file size"""
        file_path = self.file_path
        max_size = self.max_file_size_mb
        
        if file_path:
            FileSizeValidator(file_path=file_path, max_size_mb=max_size)
        
        return self


class LLMConfigInput(BaseModel):
    """Validated LLM configuration"""
    model: str = Field(default="gpt-4o", pattern=r'^[a-zA-Z0-9\-\.]+$')
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=32000)
    top_k: int = Field(default=1, ge=1, le=100)
    
    @field_validator('model')
    @classmethod
    def validate_model_name(cls, v):
        """Validate model name format"""
        # Only allow known OpenAI model patterns
        allowed_patterns = [
            r'^gpt-3\.5-turbo',
            r'^gpt-4o',
            r'^text-embedding-',
        ]
        
        if not any(re.match(pattern, v) for pattern in allowed_patterns):
            raise ValueError(f"Unsupported or invalid model name: {v}")
        
        return v


class APIKeyInput(BaseModel):
    """Validated API key (minimal validation, mainly for format)"""
    api_key: str = Field(..., min_length=20, max_length=200)
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        """Basic API key format validation"""
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        
        # Remove whitespace
        v = v.strip()
        
        # Check for obvious invalid patterns
        if ' ' in v or '\n' in v or '\t' in v:
            raise ValueError("API key contains invalid whitespace")
        
        # OpenAI keys start with 'sk-'
        if not v.startswith('sk-'):
            raise ValueError("Invalid OpenAI API key format (must start with 'sk-')")
        
        return v


def sanitize_for_logging(data: Any, max_length: int = 100) -> str:
    """
    Sanitize data for safe logging (remove sensitive info, truncate).
    
    Args:
        data: Data to sanitize
        max_length: Maximum length of output
        
    Returns:
        Sanitized string safe for logging
    """
    if data is None:
        return "None"
    
    # Convert to string
    text = str(data)
    
    # Remove potential API keys (sk-...)
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', 'sk-***REDACTED***', text)
    
    # Remove potential passwords
    text = re.sub(r'password["\']?\s*[:=]\s*["\']?[^"\'}\s]+', 'password=***REDACTED***', text, flags=re.IGNORECASE)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "...[truncated]"
    
    return text


def validate_output_path(output_path: str, workspace_root: Optional[str] = None) -> Path:
    """
    Validate output file path for writing results.
    
    Args:
        output_path: Path to validate
        workspace_root: Optional workspace root to restrict writes
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or unsafe
    """
    if not output_path or not output_path.strip():
        raise ValueError("Output path cannot be empty")
    
    # Remove dangerous characters
    output_path = output_path.replace('\x00', '')
    
    # Check for path traversal
    if '..' in output_path:
        raise ValueError("Path traversal not allowed in output path")
    
    try:
        path = Path(output_path).resolve()
        
        # Ensure parent directory exists or can be created
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # If workspace root provided, ensure output is within it
        if workspace_root:
            workspace_abs = Path(workspace_root).resolve()
            try:
                path.relative_to(workspace_abs)
            except ValueError:
                raise ValueError(f"Output path must be within workspace: {workspace_abs}")
        
        return path
        
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid output path: {str(e)}")
