"""
Field Mapping Node - Step 4 of workflow.
Maps CSV columns to multiple database tables using intelligent multi-table mapping.
"""
import logging
import re
from langchain_core.messages import HumanMessage
from ..models.workflow_state import WorkflowState
from ..services.mapper import Mapper


def field_mapping_node(state: WorkflowState) -> WorkflowState:
    """
    STEP 4: Map CSV columns to multiple database tables.
    Uses unified Mapper for multi-table field mapping.
    """
    logging.info("STEP 4: Mapping fields to database table...")
    try:
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
                'fields': match.metadata.get('fields', []),
                'metadata': match.metadata if match.metadata else {},
                'composite_score': match.similarity_score
            })
        # Prioritize selected table by moving it to the front
        if selected_table:
            logging.info(f"Prioritizing selected table: {selected_table}")
            candidate_tables.sort(
                key=lambda t: 0 if t['table_name'] == selected_table else 1
            )
        logging.info(f"Using unified Mapper with {len(candidate_tables)} candidate tables")
        if selected_table:
            logging.info(f"Primary table for mapping: {selected_table}")
        # Use Mapper for multi-table mapping
        mapper = Mapper()
        mapping_result = mapper.map_to_multiple_tables(
            file_analysis=file_analysis,
            candidate_tables=candidate_tables,
            max_tables=15
        )
        # Log results
        mapped_count = len([m for m in mapping_result.mappings if m.match_type != 'UNMAPPED'])
        tables_mapped = set([m.target_column for m in mapping_result.mappings if m.target_column])
        logging.info(f"Mapping complete: {mapped_count}/{file_analysis.structure.total_columns} columns mapped")
        logging.info(f"Tables used: {', '.join(tables_mapped)}")
        unmapped_columns = [m.source_column for m in mapping_result.mappings if m.match_type == 'UNMAPPED']
        if unmapped_columns:
            logging.warning(f"Unmapped columns: {len(unmapped_columns)}")
            logging.warning(f"    {', '.join(unmapped_columns[:5])}")
        next_step = "validation"
        workflow_status = "in_progress"
        # Create summary message
        return {
            **state,
            "field_mapping_result": mapping_result,
            "multi_table_mapping": True,
            "workflow_step": next_step,
            "workflow_status": workflow_status,
            "steps_completed": state.get("steps_completed", []) + ["field_mapping"],
            "messages": state.get("messages", []) + [
                HumanMessage(
                    content=f"Mapped {mapped_count} fields across {len(tables_mapped)} tables: {', '.join(tables_mapped)}"
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
