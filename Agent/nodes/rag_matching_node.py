import os
import logging
from ..models.workflow_state import WorkflowState
from ..services.table_matcher import TableMatcherService
from langchain_core.messages import HumanMessage

def rag_matching_node(state: WorkflowState) -> WorkflowState:
    """
    STEP 2: Search for matching database tables using RAG.
    
    Uses FileAnalysisResult from state to find best matching tables in the database schema via semantic search
    """
    logging.info("STEP 2: Searching for matching database tables...")

    try:
        file_analysis = state["file_analysis_result"]

        if not file_analysis or not file_analysis.analysis_success:
            logging.error("Cannot perform RAG matching - file analysis failed or missing.")
            return {
                **state,
                "last_error": "Cannot perform RAG matching - file analysis failed or missing.",
                "errors": state.get("errors", []) + ["RAG matching skipped due to failed file analysis."],
                "workflow_step": "error",
                "workflow_status": "failed"
            }
        
        user_context = state.get("user_context")

        service = TableMatcherService()
        match_result = service.find_matching_tables(file_analysis, user_context)

        if not match_result.matched_tables:
            logging.warning("No matching tables found in database schema.")
            return { 
                **state,
                "last_error": "No matching tables found in database schema.",
                "errors": state.get("errors", []) + ["RAG matching found no relevant tables."],
                "workflow_step": "error",
                "workflow_status": "failed"
            }
        
        logging.info(f"Found {len(match_result.matched_tables)} potential matches.")
        if match_result.primary_match:
            logging.info(f"Best match: {match_result.primary_match.table_name}")
            logging.info(f"  Score: {match_result.primary_match.similarity_score:.2f}")
            logging.info(f"  Confidence: {match_result.primary_match.confidence}")

        logging.info(f"Top matches:")
        for i, match in enumerate(match_result.matched_tables[:1], 1):
            logging.info(f"  {i}. {match.table_name} (score: {match.similarity_score:.2f}, confidence: {match.confidence})")

        basename = os.path.basename(state["file_path"])
        return {
            **state,
            "rag_match_result": match_result,
            "workflow_step": "table_selection",
            "steps_completed": state.get("steps_completed", []) + ["rag_matching"],
            "messages": state.get("messages", []) + [
                HumanMessage(
                    content=f"Found {len(match_result.matched_tables)} matching tables for {basename}."
                )
            ]
        }
    except Exception as e:
        error_msg = f"Exception in RAG matching: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return {
            **state,
            "last_error": error_msg,
            "errors": state.get("errors", []) + [error_msg],
            "workflow_step": "error",
            "workflow_status": "failed"
        }


