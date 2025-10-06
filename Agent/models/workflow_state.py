"""
WorkflowState TypedDict - Core state management for the workflow.

Extracted from Agent_old/agent.py to provide a clean, typed state definition.
"""

from typing import Optional, TypedDict
from .file_analysis_model import FileAnalysisResult
from .rag_match_model import TableMatchResult, FieldMappingResult

class WorkflowState(TypedDict, total=False):
    """
    State object passed between workflow nodes.
    
    Attributes:
        messages: List of conversation messages
        file_path: Path to the file being analyzed
        user_context: Optional user-provided context
        table_preference: Optional preferred table name
        user_preferences: Optional user preferences string
        
        file_analysis_result: Result from file analysis step
        rag_results: Results from RAG matching
        selected_table: Selected table and mappings
        validation_result: Validation results
        
        workflow_step: Current workflow step name
        validation_status: Current validation status
        refinement_attempts: Number of refinement attempts made
        max_refinements: Maximum allowed refinements
        workflow_status: Overall workflow status
        errors: List of error messages
        last_error: Most recent error message
        steps_completed: List of completed step names
        refinement_history: History of refinement attempts
    """
    
    # Input fields
    messages: list
    file_path: str
    table_preference: Optional[str]
    user_preferences: Optional[str]
    

    # rag_results: Optional[dict]
    selected_table: Optional[dict]
    validation_result: Optional[dict]
    
    # Workflow control
    workflow_step: str
    validation_status: Optional[str]
    refinement_attempts: int
    max_refinements: int
    workflow_status: str
    
    # Error tracking
    errors: list
    last_error: Optional[str]
    
    # Progress tracking
    steps_completed: list
    refinement_history: list
    
    # Optional final outputs
    mapping_visualization: Optional[dict]
    ready_for_execution: Optional[bool]
    ready_for_review: Optional[bool]
    error_analysis: Optional[dict]

    # File analysis results
    file_analysis_result: Optional[FileAnalysisResult]

    # RAG matching (Step 2)
    rag_match_result: Optional[TableMatchResult]
    
    # Table selection (Step 3)
    selected_table: Optional[str]
    selected_schema: Optional[str]
    selected_table_metadata: Optional[dict]
    
    # Field mapping (Step 4)
    field_mapping_result: Optional[FieldMappingResult]

    # Optional
    user_context: Optional[str]
