from enum import Enum 

class WorkflowStatus(str, Enum):
    VALIDATION_PASSED = "validation_passed"
    REQUIRES_MANUAL_REVIEW = "requires_manual_review"
    FAILED = "failed" 

class FileType(str, Enum):
    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    CSV = "csv"

# Default values
DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_REFINEMENTS = 3
DEFAULT_CONFIDENCE_THRESHOLD =0.7 

DEFAULT_RAG_COLLECTION = "maxo_vector_store_v2"