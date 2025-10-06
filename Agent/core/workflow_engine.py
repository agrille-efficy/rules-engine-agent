"""
Workflow Engine - Executes the LangGraph workflow.
Main orchestrator for file-to-database mapping.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..models.workflow_state import WorkflowState
from .graph_builder import WorkflowGraphBuilder


class WorkflowEngine:
    """
    Main workflow execution engine.
    Orchestrates the complete file-to-database mapping process.
    """
    
    def __init__(self, use_checkpointer: bool = True):
        """
        Initialize workflow engine.
        
        Args:
            use_checkpointer: Whether to use state checkpointing for recovery
        """
        self.builder = WorkflowGraphBuilder(use_checkpointer=use_checkpointer)
        self.graph = None
        self.execution_history = []
        
    def initialize(self):
        """Build and compile the workflow graph."""
        logging.info("Initializing workflow engine...")
        self.graph = self.builder.build()
        logging.info("Workflow engine ready!")
        
    def run(
        self,
        file_path: str,
        user_context: Optional[str] = None,
        table_preference: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the complete workflow for a given file.
        
        Args:
            file_path: Path to the CSV file to process
            user_context: Optional user-provided context
            table_preference: Optional preferred table name
            **kwargs: Additional workflow parameters
            
        Returns:
            Final workflow state as dictionary
        """
        # Initialize graph if not already done
        if self.graph is None:
            self.initialize()
        
        # Create initial state
        initial_state: WorkflowState = {
            "file_path": file_path,
            "workflow_step": "start",
            "workflow_status": "in_progress",
            "steps_completed": [],
            "messages": [],
            "errors": [],
            "user_context": user_context,
            "table_preference": table_preference,
            **kwargs
        }
        
        logging.info("=" * 80)
        logging.info("STARTING WORKFLOW EXECUTION")
        logging.info("=" * 80)
        logging.info(f"File: {file_path}")
        if user_context:
            logging.info(f"Context: {user_context}")
        if table_preference:
            logging.info(f"Table preference: {table_preference}")
        
        start_time = datetime.now()
        
        try:
            # Execute workflow
            final_state = self._execute_workflow(initial_state)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log summary
            self._log_execution_summary(final_state, execution_time)
            
            # Store in history
            self.execution_history.append({
                "file_path": file_path,
                "start_time": start_time,
                "execution_time": execution_time,
                "status": final_state.get("workflow_status"),
                "final_step": final_state.get("workflow_step"),
                "steps_completed": final_state.get("steps_completed", [])
            })
            
            return dict(final_state)
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logging.error(error_msg, exc_info=True)
            
            return {
                **initial_state,
                "workflow_status": "failed",
                "workflow_step": "error",
                "last_error": error_msg,
                "errors": [error_msg]
            }
    
    def _execute_workflow(self, initial_state: WorkflowState) -> WorkflowState:
        """
        Execute the workflow graph.
        
        Args:
            initial_state: Initial workflow state
            
        Returns:
            Final workflow state
        """
        # Stream execution for step-by-step processing
        final_state = None
        
        # Configure execution with thread_id for checkpointing
        config = {
            "configurable": {
                "thread_id": f"workflow_{initial_state['file_path'].replace('/', '_').replace('\\', '_')}"
            }
        }
        
        for step_output in self.graph.stream(initial_state, config):
            # Each step_output is a dict with node name as key
            for node_name, node_state in step_output.items():
                logging.info(f"Completed node: {node_name}")
                final_state = node_state
        
        return final_state
    
    def _log_execution_summary(self, final_state: WorkflowState, execution_time: float):
        """
        Log a summary of workflow execution.
        
        Args:
            final_state: Final state after workflow completion
            execution_time: Total execution time in seconds
        """
        logging.info("=" * 80)
        logging.info("WORKFLOW EXECUTION SUMMARY")
        logging.info("=" * 80)
        
        status = final_state.get("workflow_status", "unknown")
        final_step = final_state.get("workflow_step", "unknown")
        steps_completed = final_state.get("steps_completed", [])
        errors = final_state.get("errors", [])
        
        logging.info(f"Status: {status}")
        logging.info(f"Final step: {final_step}")
        logging.info(f"Steps completed: {len(steps_completed)}")
        logging.info(f"Execution time: {execution_time:.2f}s")
        
        if steps_completed:
            logging.info(f"Completed steps: {' → '.join(steps_completed)}")
        
        if errors:
            logging.warning(f"Errors encountered: {len(errors)}")
        
        # Show key results
        if final_state.get("file_analysis_result"):
            result = final_state["file_analysis_result"]
            logging.info(f"✓ File analyzed: {result.structure.total_rows} rows, {result.structure.total_columns} columns")
        
        if final_state.get("rag_match_result"):
            result = final_state["rag_match_result"]
            logging.info(f"✓ RAG matches: {len(result.matched_tables)} tables found")
        
        if final_state.get("selected_table"):
            table = final_state["selected_table"]
            metadata = final_state.get("selected_table_metadata", {})
            confidence = metadata.get("confidence", "unknown")
            logging.info(f"✓ Selected table: {table} ({confidence} confidence)")
        
        if final_state.get("field_mapping_result"):
            result = final_state["field_mapping_result"]
            
            # Check if multi-table mapping (has 'table_mappings' attribute)
            if hasattr(result, 'table_mappings'):
                # Multi-table mapping result
                total_mapped = sum(len(tm.mappings) for tm in result.table_mappings)
                logging.info(f"✓ Multi-table mapping: {total_mapped}/{result.total_source_columns} columns across {len(result.table_mappings)} tables "
                            f"({result.overall_coverage:.1f}%, {result.overall_confidence} confidence)")
            else:
                # Single-table mapping result
                validation = result.validation
                logging.info(f"✓ Field mapping: {validation.mapped_count}/{validation.total_mappings} columns "
                            f"({validation.mapping_coverage_percent:.1f}%, {validation.confidence_level} confidence)")
        
        logging.info("=" * 80)
    
    def get_execution_history(self) -> list:
        """Get history of all workflow executions."""
        return self.execution_history
    
    def visualize_workflow(self, output_path: Optional[str] = None):
        """Visualize the workflow graph structure."""
        self.builder.visualize(output_path)
