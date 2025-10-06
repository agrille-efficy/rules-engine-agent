"""
Field Mapping Node - Step 4 of workflow.
Maps CSV columns to multiple database tables using intelligent multi-table mapping.
"""
import logging
import re
from langchain_core.messages import HumanMessage
from ..models.workflow_state import WorkflowState
from ..services.multi_table_mapper import MultiTableFieldMapper


def field_mapping_node(state: WorkflowState) -> WorkflowState:
    """
    STEP 4: Map CSV columns to multiple database tables.
    
    Uses intelligent multi-table field mapping with semantic grouping.
    Can map data to multiple tables simultaneously (e.g., Opportunity + related tables).
    """
    logging.info("STEP 4: Mapping fields to database table...")
    
    try:
        # Get required data from state
        file_analysis = state.get("file_analysis_result")
        rag_match_result = state.get("rag_match_result")
        selected_table = state.get("selected_table")
        
        if not file_analysis or not rag_match_result:
            logging.error("Missing file analysis or RAG match result")
            return {
                **state,
                "last_error": "Cannot map fields - missing analysis or RAG matches",
                "errors": state.get("errors", []) + ["Field mapping failed: missing data"],
                "workflow_step": "error",
                "workflow_status": "failed"
            }
        
        # Convert RAG matches to candidate tables format
        candidate_tables = []
        for match in rag_match_result.matched_tables:
            candidate_tables.append({
                'table_name': match.table_name,
                'table_kind': match.metadata.get('table_kind', 'Entity') if match.metadata else 'Entity',
                'table_code': match.metadata.get('table_code', '') if match.metadata else '',
                'composite_score': match.similarity_score,
                'field_count': match.metadata.get('field_count', 0) if match.metadata else 0,
                'query_coverage': match.metadata.get('query_coverage', 0) if match.metadata else 0,
            })
        
        logging.info(f"Using multi-table mapper with {len(candidate_tables)} candidate tables")
        
        # Initialize multi-table mapper and perform mapping
        mapper = MultiTableFieldMapper()
        mapping_result = mapper.map_to_multiple_tables(
            file_analysis,
            candidate_tables,
            max_tables=5  # Map to up to 5 tables
        )
        
        # Log comprehensive results
        logging.info(f"Mapping complete:")
        logging.info(f"  Mapped: {sum(len(tm.mappings) for tm in mapping_result.table_mappings)}/{mapping_result.total_source_columns} columns")
        logging.info(f"  Coverage: {mapping_result.overall_coverage:.1f}%")
        logging.info(f"  Confidence: {mapping_result.overall_confidence}")
        logging.info(f"  Valid: {mapping_result.is_valid}")
        logging.info(f"  Tables used: {len(mapping_result.table_mappings)}")
        
        # Log per-table breakdown
        for tm in mapping_result.table_mappings:
            logging.info(f"    → {tm.table_name}: {len(tm.mappings)} columns ({tm.validation.mapping_coverage_percent:.1f}%, {tm.validation.confidence_level})")
        
        if mapping_result.unmapped_columns:
            logging.warning(f"  Unmapped columns: {len(mapping_result.unmapped_columns)}")
            logging.warning(f"    {', '.join(mapping_result.unmapped_columns[:5])}")
        
        # Determine next step based on validation
        if not mapping_result.is_valid or mapping_result.requires_review:
            next_step = "validation"
            workflow_status = "requires_review"
            logging.warning("Mapping requires review")
        else:
            next_step = "validation"
            workflow_status = "in_progress"
            logging.info("Mapping successful, proceeding to validation")
        
        # Create summary message
        tables_mapped = ", ".join([tm.table_name for tm in mapping_result.table_mappings])
        total_mapped = sum(len(tm.mappings) for tm in mapping_result.table_mappings)
        
        return {
            **state,
            "field_mapping_result": mapping_result,
            "multi_table_mapping": True,  # Flag to indicate multi-table mapping was used
            "workflow_step": next_step,
            "workflow_status": workflow_status,
            "steps_completed": state.get("steps_completed", []) + ["field_mapping"],
            "messages": state.get("messages", []) + [
                HumanMessage(
                    content=f"Mapped {total_mapped} fields across {len(mapping_result.table_mappings)} tables: {tables_mapped} "
                           f"({mapping_result.overall_confidence} confidence)"
                )
            ]
        }
        
    except Exception as e:
        error_msg = f"Exception in field mapping: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return {
            **state,
            "last_error": error_msg,
            "errors": state.get("errors", []) + [error_msg],
            "workflow_step": "error",
            "workflow_status": "failed"
        }


def _extract_table_fields(table_metadata: dict) -> list:
    """
    Extract field names from table metadata.
    
    Tries to parse from RAG content or metadata.
    """
    fields = []
    
    # Try to get from metadata
    metadata = table_metadata.get("metadata", {})
    
    # Check if fields are in metadata
    if "fields" in metadata:
        fields_data = metadata["fields"]
        if isinstance(fields_data, list):
            fields = fields_data
        elif isinstance(fields_data, str):
            # Parse from string
            fields = [f.strip() for f in fields_data.split(",")]
    
    # Try parsing from content (fallback)
    if not fields:
        content = metadata.get("content", "")
        if content:
            # Look for field patterns in content
            # Example: "Primary fields: K_MAIL, MAIL_EXPEDITEUR, MAIL_CC"
            field_match = re.search(r'Primary fields?:\s*([^\n]+)', content)
            if field_match:
                field_str = field_match.group(1)
                fields = [f.strip() for f in field_str.split(',') if f.strip()]
            else:
                # Try to extract field-like patterns (uppercase with underscores)
                field_pattern = r'\b[A-Z][A-Z0-9_]{2,}\b'
                potential_fields = re.findall(field_pattern, content)
                # Remove duplicates while preserving order
                seen = set()
                fields = [f for f in potential_fields if f not in seen and not seen.add(f)]
    
    return fields[:50]  # Limit to 50 fields max


def _generate_placeholder_fields(table_name: str) -> list:
    """
    Generate placeholder field names when real schema is unavailable.
    
    This is a temporary solution until DICO API integration.
    """
    # Common field patterns by table type
    common_fields = [f"K_{table_name.upper()}"]  # Primary key
    
    # Add common business fields
    common_fields.extend([
        f"{table_name.upper()}_NAME",
        f"{table_name.upper()}_DATE",
        f"{table_name.upper()}_STATUS",
        f"{table_name.upper()}_DESCRIPTION",
        f"{table_name.upper()}_AMOUNT",
        f"{table_name.upper()}_TYPE",
        "CREATED_DATE",
        "MODIFIED_DATE",
        "CREATED_BY",
        "MODIFIED_BY"
    ])
    
    logging.info(f"Generated {len(common_fields)} placeholder fields for {table_name}")
    return common_fields
