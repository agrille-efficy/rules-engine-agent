"""
Routing Logic - Determines next step in workflow.
Handles conditional branching based on state.
"""
import logging
from typing import Literal
from ..models.workflow_state import WorkflowState


def route_after_file_analysis(
    state: WorkflowState
) -> Literal["rag_matching", "error"]:
    """
    Route after file analysis step.
    
    Returns:
        "rag_matching" if analysis successful
        "error" if analysis failed
    """
    if state.get("workflow_step") == "error":
        logging.error("File analysis failed, routing to error")
        return "error"
    
    if not state.get("file_analysis_result"):
        logging.error("No file analysis result, routing to error")
        return "error"
    
    logging.info("File analysis successful, routing to RAG matching")
    return "rag_matching"


def route_after_rag_matching(
    state: WorkflowState
) -> Literal["table_selection", "error"]:
    """
    Route after RAG matching step.
    
    Returns:
        "table_selection" if matches found
        "error" if no matches or error
    """
    if state.get("workflow_step") == "error":
        logging.error("RAG matching failed, routing to error")
        return "error"
    
    rag_result = state.get("rag_match_result")
    if not rag_result or not rag_result.matched_tables:
        logging.error("No RAG matches found, routing to error")
        return "error"
    
    logging.info(f"RAG matching successful ({len(rag_result.matched_tables)} matches), routing to table selection")
    return "table_selection"


def route_after_table_selection(
    state: WorkflowState
) -> Literal["field_mapping", "error"]:
    """
    Route after table selection step.
    
    Returns:
        "field_mapping" if table selected
        "error" if selection failed
    """
    if state.get("workflow_step") == "error":
        logging.error("Table selection failed, routing to error")
        return "error"
    
    if not state.get("selected_table"):
        logging.error("No table selected, routing to error")
        return "error"
    
    logging.info(f"Table selection successful ({state['selected_table']}), routing to field mapping")
    return "field_mapping"


def route_after_field_mapping(
    state: WorkflowState
) -> Literal["validation", "review", "error", "end"]:
    """
    Route after field mapping step.
    
    Supports both single-table and multi-table mapping results.
    
    Returns:
        "validation" if mapping successful and valid
        "review" if mapping needs human review
        "error" if mapping failed
        "end" if complete (for now, since validation node not built yet)
    """
    if state.get("workflow_step") == "error":
        logging.error("Field mapping failed, routing to error")
        return "error"
    
    mapping_result = state.get("field_mapping_result")
    if not mapping_result:
        logging.error("No field mapping result, routing to error")
        return "error"
    
    # Check if multi-table mapping (has 'table_mappings' attribute)
    if hasattr(mapping_result, 'table_mappings'):
        # Multi-table mapping result
        is_valid = mapping_result.is_valid
        requires_review = mapping_result.requires_review
        coverage = mapping_result.overall_coverage
        
        logging.info(f"Multi-table mapping: {len(mapping_result.table_mappings)} tables, {coverage:.1f}% coverage")
    else:
        # Single-table mapping result
        validation = mapping_result.validation
        is_valid = validation.is_valid
        requires_review = validation.requires_review
        coverage = validation.mapping_coverage_percent
        
        # logging.info(f"Single-table mapping: {coverage:.1f}% coverage")
    
    # Check if requires review
    if requires_review or not is_valid:
        logging.warning(f"Field mapping requires review (valid: {is_valid}, coverage: {coverage:.1f}%)")
        return "review"
    
    # For now, end here since validation node not built yet
    # Later: return "validation"
    logging.info("Field mapping successful, workflow complete")
    return "end"


def route_from_error(
    state: WorkflowState
) -> Literal["end"]:
    """
    Route from error state.
    
    For now, just end the workflow.
    Future: Could implement retry logic or human intervention.
    """
    logging.error(f"Workflow ended in error state: {state.get('last_error')}")
    return "end"


def route_from_review(
    state: WorkflowState
) -> Literal["end"]:
    """
    Route from review state.
    
    For now, just end the workflow.
    Future: Could wait for human review and continue.
    """
    logging.info("Workflow requires human review, ending for now")
    return "end"


def should_continue_workflow(
    state: WorkflowState
) -> bool:
    """
    Determine if workflow should continue.
    
    Returns:
        True if workflow can continue
        False if workflow should stop
    """
    workflow_status = state.get("workflow_status", "")
    workflow_step = state.get("workflow_step", "")
    
    # Stop conditions
    if workflow_status == "failed":
        return False
    
    if workflow_step in ["error", "end"]:
        return False
    
    # Check for max errors
    errors = state.get("errors", [])
    if len(errors) >= 5:
        logging.error("Too many errors (5+), stopping workflow")
        return False
    
    return True
