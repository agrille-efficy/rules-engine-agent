
import os
from langchain_core.messages import HumanMessage
from ..models.workflow_state import WorkflowState
from ..services.file_analyzer import FileAnalyzerService


def file_analysis_node(state: WorkflowState) -> WorkflowState:
    """
    STEP 1: Analyze file structure and content
    Calls the analyze_file tool from tools.py
    """
    file_path = state['file_path']
    print(f"STEP 1: Analyzing file structure and content for "
          f"{file_path}...")

    try:
        service = FileAnalyzerService()
        result = service.analyze(state['file_path'])

        if not result.analysis_success:
            return {
                **state,
                "last_error": result.error_message,
                "errors": (
                    state.get("errors", []) +
                    [f"File Analysis: {result.error_message}"]
                ),
                "workflow_step": "error",
                "workflow_status": "failed"
            }

        print(f"  File analyzed successfully")
        print(f"    Rows: {result.structure.total_rows}")
        print(f"    Columns: {result.structure.total_columns}")

        basename = os.path.basename(state['file_path'])
        return {
            **state,
            "file_analysis_result": result,
            "workflow_step": "rag_matching",
            "steps_completed": (
                state.get("steps_completed", []) + ["file_analysis"]
            ),
            "messages": state.get("messages", []) + [
                HumanMessage(
                    content=f"File analysis completed for {basename}"
                )
            ]
        }

    except Exception as e:
        error_msg = f"Exception in file analysis: {str(e)}"
        print(f"  ERROR: {error_msg}")
        return {
            **state,
            "last_error": error_msg,
            "errors": state.get("errors", []) + [error_msg],
            "workflow_step": "error",
            "workflow_status": "failed"
        }