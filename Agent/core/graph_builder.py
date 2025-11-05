"""
Graph Builder - Constructs the LangGraph workflow.
Defines nodes, edges, and routing logic.
"""
import logging
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..models.workflow_state import WorkflowState
from ..nodes.file_analysis_node import file_analysis_node
from ..nodes.rag_matching_node import rag_matching_node
from ..nodes.table_selection_node import table_selection_node
from ..nodes.field_mapping_node import field_mapping_node
from ..routing.routing_logic import (
    route_after_file_analysis,
    route_after_rag_matching,
    route_after_table_selection,
    route_after_field_mapping
)


class WorkflowGraphBuilder:
    """
    Builds the LangGraph workflow for file-to-database mapping.
    """
    
    def __init__(self, use_checkpointer: bool = True):
        """
        Initialize graph builder.
        
        Args:
            use_checkpointer: Whether to use memory checkpointer for state persistence
        """
        self.use_checkpointer = use_checkpointer
        self.checkpointer = MemorySaver() if use_checkpointer else None
        
    def build(self) -> StateGraph:
        """
        Build and compile the workflow graph.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        logging.info("Building workflow graph...")
        
        # Initialize graph with WorkflowState
        graph = StateGraph(WorkflowState)
        
        # Add nodes
        logging.info("Adding nodes to graph...")
        graph.add_node("file_analysis", file_analysis_node)
        graph.add_node("rag_matching", rag_matching_node)
        graph.add_node("table_selection", table_selection_node)
        graph.add_node("field_mapping", field_mapping_node)
        graph.add_node("error", self._error_handler_node)
        graph.add_node("review", self._review_handler_node)
        
        # Set entry point
        graph.set_entry_point("file_analysis")
        
        # Add conditional edges with routing logic
        logging.info("Adding conditional edges...")
        
        # After file analysis -> rag_matching or error
        graph.add_conditional_edges(
            "file_analysis",
            route_after_file_analysis,
            {
                "rag_matching": "rag_matching",
                "error": "error"
            }
        )
        
        # After RAG matching -> table_selection or error
        graph.add_conditional_edges(
            "rag_matching",
            route_after_rag_matching,
            {
                "table_selection": "table_selection",
                "error": "error"
            }
        )
        
        # After table selection -> field_mapping or error
        graph.add_conditional_edges(
            "table_selection",
            route_after_table_selection,
            {
                "field_mapping": "field_mapping",
                "error": "error"
            }
        )
        
        # After field mapping -> validation/review/error/end
        graph.add_conditional_edges(
            "field_mapping",
            route_after_field_mapping,
            {
                "validation": END,  # Will be validation node in future
                "review": "review",
                "error": "error",
                "end": END
            }
        )
        
        # Error and review nodes end the workflow
        graph.add_edge("error", END)
        graph.add_edge("review", END)
        
        # Compile graph
        logging.info("Compiling graph...")
        compiled_graph = graph.compile(
            checkpointer=self.checkpointer,
            interrupt_before=None,  # No human-in-the-loop interrupts for now
            interrupt_after=None
        )
        
        logging.info("Graph compilation complete!")
        return compiled_graph
    
    def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """
        Handle error state.
        Logs error details and marks workflow as failed.
        """
        logging.error("=" * 80)
        logging.error("WORKFLOW ERROR HANDLER")
        logging.error("=" * 80)
        
        last_error = state.get("last_error", "Unknown error")
        errors = state.get("errors", [])
        
        logging.error(f"Last error: {last_error}")
        logging.error(f"Total errors: {len(errors)}")
        
        if errors:
            logging.error("Error history:")
            for i, error in enumerate(errors, 1):
                logging.error(f"  {i}. {error}")
        
        return {
            **state,
            "workflow_step": "error",
            "workflow_status": "failed"
        }
    
    def _review_handler_node(self, state: WorkflowState) -> WorkflowState:
        """
        Handle review state.
        Logs what needs review and marks workflow for human attention.
        Supports both single-table and multi-table mapping results.
        """
        logging.warning("=" * 80)
        logging.warning("WORKFLOW REQUIRES REVIEW")
        logging.warning("=" * 80)
        
        mapping_result = state.get("field_mapping_result")
        if mapping_result:
            # Check if multi-table mapping (has 'table_mappings' attribute)
            if hasattr(mapping_result, 'table_mappings'):
                # Multi-table mapping result
                logging.warning(f"Mapping coverage: {mapping_result.overall_coverage:.1f}%")
                logging.warning(f"Confidence level: {mapping_result.overall_confidence}")
                logging.warning(f"Valid: {mapping_result.is_valid}")
                
                if mapping_result.unmapped_columns:
                    logging.warning("Issues requiring attention:")
                    logging.warning(f"  - {len(mapping_result.unmapped_columns)} unmapped columns")
                
                if mapping_result.unmapped_columns:
                    logging.warning("Warnings:")
                    for col in mapping_result.unmapped_columns[:5]:
                        logging.warning(f"  - {col}")
            else:
                # Single-table mapping result
                validation = mapping_result.validation
                
                logging.warning(f"Mapping coverage: {validation.mapping_coverage_percent:.1f}%")
                logging.warning(f"Confidence level: {validation.confidence_level}")
                logging.warning(f"Valid: {validation.is_valid}")
                
                if validation.issues:
                    logging.warning("Issues requiring attention:")
                    for issue in validation.issues:
                        logging.warning(f"  - {issue}")
                
                if validation.warnings:
                    logging.warning("Warnings:")
                    for warning in validation.warnings[:5]:
                        logging.warning(f"  - {warning}")
        
        logging.warning("Workflow paused for human review")
        
        return {
            **state,
            "workflow_step": "review",
            "workflow_status": "requires_review"
        }
    
    def visualize(self, output_path: Optional[str] = None) -> None:
        """
        Generate a visual representation of the workflow graph.
        
        Args:
            output_path: Optional path to save the visualization
        """
        try:
            graph = self.build()
            
            # # Try to generate ASCII representation
            # logging.info("Workflow Graph Structure:")
            # logging.info("=" * 80)
            # logging.info("START")
            # logging.info("  ↓")
            # logging.info("file_analysis")
            # logging.info("  ├─→ rag_matching (if success)")
            # logging.info("  └─→ error (if failed)")
            # logging.info("      ↓")
            # logging.info("rag_matching")
            # logging.info("  ├─→ table_selection (if matches found)")
            # logging.info("  └─→ error (if no matches)")
            # logging.info("      ↓")
            # logging.info("table_selection")
            # logging.info("  ├─→ field_mapping (if table selected)")
            # logging.info("  └─→ error (if failed)")
            # logging.info("      ↓")
            # logging.info("field_mapping")
            # logging.info("  ├─→ END (if valid)")
            # logging.info("  ├─→ review (if needs review)")
            # logging.info("  └─→ error (if failed)")
            # logging.info("=" * 80)
            
            if output_path:
                logging.info(f"Graph visualization would be saved to: {output_path}")
                
        except Exception as e:
            logging.error(f"Failed to visualize graph: {e}")
