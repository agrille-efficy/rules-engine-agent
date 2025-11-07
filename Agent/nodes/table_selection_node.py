"""
Table Selection Node - Step 3 of workflow.
Selects the best matching table from RAG results.
"""
import logging
from langchain_core.messages import HumanMessage
from ..models.workflow_state import WorkflowState


def table_selection_node(state: WorkflowState) -> WorkflowState:
    """
    STEP 3: Select the best matching database table.
    
    For now, auto-selects the highest confidence match.
    Future: Can add user interaction/confirmation here.
    """
    logging.info("STEP 3: Selecting database table...")
    
    try:
        # Get RAG match results from state
        rag_result = state.get("rag_match_result")
        
        if not rag_result or not rag_result.matched_tables:
            logging.error("No RAG match results found in state")
            return {
                **state,
                "last_error": "Cannot select table - no matches found",
                "errors": state.get("errors", []) + ["Table selection failed: no matches"],
                "workflow_step": "error",
                "workflow_status": "failed"
            }
        
        # Get user preference if provided
        user_preference = state.get("table_preference")
        
        # Selection logic
        selected_match = None
        selection_reason = ""
        
        if user_preference:
            # Check if user's preferred table is in the results
            for match in rag_result.matched_tables:
                if match.table_name.lower() == user_preference.lower():
                    selected_match = match
                    selection_reason = f"User preference: {user_preference}"
                    logging.info(f"Selected user-preferred table: {user_preference}")
                    break
            
            if not selected_match:
                logging.warning(f"User preference '{user_preference}' not found in matches, using best match")
        
        # If no user preference or preference not found, use smart selection
        if not selected_match:
            # Smart selection: Prefer Entity tables over Relationship tables for primary table
            entity_tables = []
            relation_tables = []
            
            for match in rag_result.matched_tables:
                table_kind = match.metadata.get('table_kind', 'Entity') if match.metadata else 'Entity'
                if table_kind == 'Entity':
                    entity_tables.append(match)
                else:
                    relation_tables.append(match)
            
            # Prefer the highest-scoring Entity table as primary
            if entity_tables:
                selected_match = entity_tables[0]  # Already sorted by score
                selection_reason = f"Best Entity table (score: {selected_match.similarity_score:.2f})"
                logging.info(f"Auto-selected best Entity table: {selected_match.table_name}")
            else:
                # Fallback to highest score overall (even if it's a relationship table)
                selected_match = rag_result.primary_match
                selection_reason = f"Highest confidence match (score: {selected_match.similarity_score:.2f})"
                logging.info(f"Auto-selected best match: {selected_match.table_name}")
                logging.warning("No Entity tables found, selected a Relationship table as primary")
        
        # Log selection details
        logging.info(f"Selected table: {selected_match.table_name}")
        logging.info(f"  Confidence: {selected_match.confidence}")
        logging.info(f"  Score: {selected_match.similarity_score:.3f}")
        logging.info(f"  Reason: {selection_reason}")
        
        # Check if selection meets minimum confidence threshold
        min_confidence_threshold = 0.5
        if selected_match.similarity_score < min_confidence_threshold:
            logging.warning(f"Selected table has low confidence: {selected_match.similarity_score:.2f}")
            return {
                **state,
                "selected_table": selected_match.table_name,
                "selected_schema": selected_match.schema_name,
                "selected_table_metadata": {
                    "table_name": selected_match.table_name,
                    "schema_name": selected_match.schema_name,
                    "confidence": selected_match.confidence,
                    "similarity_score": selected_match.similarity_score,
                    "selection_reason": selection_reason,
                    "requires_review": True,
                    "metadata": selected_match.metadata
                },
                "last_error": f"Low confidence match: {selected_match.similarity_score:.2f}",
                "errors": state.get("errors", []) + [f"Low confidence: {selected_match.similarity_score:.2f}"],
                "workflow_step": "validation",  # Proceed but flag for review
                "workflow_status": "requires_review",
                "steps_completed": state.get("steps_completed", []) + ["table_selection"]
            }
        
        # Success - good confidence match
        return {
            **state,
            "selected_table": selected_match.table_name,
            "selected_schema": selected_match.schema_name,
            "selected_table_metadata": {
                "table_name": selected_match.table_name,
                "schema_name": selected_match.schema_name,
                "confidence": selected_match.confidence,
                "similarity_score": selected_match.similarity_score,
                "selection_reason": selection_reason,
                "requires_review": False,
                "metadata": selected_match.metadata
            },
            "workflow_step": "field_mapping",  # Next step
            "workflow_status": "in_progress",
            "steps_completed": state.get("steps_completed", []) + ["table_selection"],
            "messages": state.get("messages", []) + [
                HumanMessage(
                    content=f"Selected table '{selected_match.table_name}' with {selected_match.confidence} confidence"
                )
            ]
        }
        
    except Exception as e:
        error_msg = f"Exception in table selection: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return {
            **state,
            "last_error": error_msg,
            "errors": state.get("errors", []) + [error_msg],
            "workflow_step": "error",
            "workflow_status": "failed"
        }
