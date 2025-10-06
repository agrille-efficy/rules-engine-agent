"""
Database Schema Service - Fetches real table schemas.
Extracts field information from RAG metadata and vector store.
"""
import logging
import re
from typing import List, Dict, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue


class DatabaseSchemaService:
    """
    Service for fetching real database table schemas.
    Integrates with RAG vector store to get actual field definitions.
    """
    
    def __init__(self):
        # Import here to avoid circular dependency
        from ..rag.pipeline import qdrant_client, embeddings
        
        self.client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = "maxo_vector_store_v2"
        
    def get_table_fields(
        self,
        table_name: str,
        schema_name: Optional[str] = None
    ) -> List[str]:
        """
        Get real fields for a database table.
        
        Args:
            table_name: Name of the database table
            schema_name: Optional schema name
            
        Returns:
            List of field names for the table
        """
        logging.info(f"Fetching schema for table: {table_name}")
        
        # Try to get fields from RAG metadata
        fields = self._extract_from_rag(table_name)
        
        if fields:
            logging.info(f"Found {len(fields)} real fields for {table_name}")
            return fields
        
        # Fallback to placeholder if no real schema found
        logging.warning(f"No real schema found for {table_name}, using placeholders")
        return self._generate_placeholder_fields(table_name)
    
    def _extract_from_rag(self, table_name: str) -> List[str]:
        """
        Extract field names from RAG vector store.
        
        Args:
            table_name: Table name to search for
            
        Returns:
            List of field names
        """
        try:
            # Search for table documentation in RAG using query_points with filter
            results = self.client.query_points(
                collection_name=self.collection_name,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="primary_table",
                            match=MatchValue(value=table_name)
                        )
                    ]
                ),
                limit=5,
                with_payload=True
            )
            
            if not results or not results.points:
                logging.debug(f"No RAG points found for table {table_name}")
                return []
            
            # Extract fields from metadata
            fields = set()
            
            for point in results.points:
                payload = point.payload
                
                # Method 1: Check for 'fields' in payload
                if 'fields' in payload:
                    field_data = payload['fields']
                    if isinstance(field_data, list):
                        fields.update(field_data)
                    elif isinstance(field_data, str):
                        # Parse comma-separated or newline-separated
                        parsed = re.split(r'[,\n]', field_data)
                        fields.update([f.strip() for f in parsed if f.strip()])
                
                # Method 2: Parse from content (improved parsing)
                if 'content' in payload:
                    content = payload['content']
                    
                    # Look for FIELD DEFINITIONS section
                    # Pattern: "- FIELD_NAME (FIELD_CODE): TYPE(SIZE) | REQUIRED/NULLABLE [PK/FK]"
                    field_def_pattern = r'^-\s+([A-Za-z0-9_]+)\s+\(([A-Za-z0-9_]+)\):\s+'
                    
                    lines = content.split('\n')
                    in_field_section = False
                    
                    for line in lines:
                        # Detect FIELD DEFINITIONS section
                        if '# FIELD DEFINITIONS' in line or 'FIELD DEFINITIONS' in line:
                            in_field_section = True
                            continue
                        
                        # Stop at next section
                        if in_field_section and line.strip().startswith('#'):
                            break
                        
                        # Parse field definition lines
                        if in_field_section:
                            match = re.match(field_def_pattern, line)
                            if match:
                                field_name = match.group(1)  # First capture group is field name
                                # Verify it's a real field (not a metadata keyword)
                                if not field_name.upper() in {'TABLE', 'FIELD', 'PRIMARY', 'KEY', 'FOREIGN', 
                                                              'INDEX', 'UNIQUE', 'ENTITY', 'RELATION', 
                                                              'ALPHANUMERIC', 'BOOLEAN', 'CONSTRAINTS', 
                                                              'DATE', 'DEFINITIONS', 'NULLABLE', 'REQUIRED'}:
                                    fields.add(field_name)
                    
                    # Fallback: Look for "Primary fields:" pattern
                    if not fields:
                        field_match = re.search(r'Primary fields?:\s*([^\n]+)', content, re.IGNORECASE)
                        if field_match:
                            field_str = field_match.group(1)
                            parsed = [f.strip() for f in field_str.split(',') if f.strip()]
                            fields.update(parsed)
            
            if fields:
                # Sort and return
                field_list = sorted(list(fields))
                logging.info(f"Extracted {len(field_list)} fields from RAG: {', '.join(field_list[:5])}...")
                return field_list
            
            return []
            
        except Exception as e:
            logging.error(f"Error extracting fields from RAG: {e}", exc_info=True)
            return []
    
    def _generate_placeholder_fields(self, table_name: str) -> List[str]:
        """
        Generate placeholder field names when real schema unavailable.
        
        Args:
            table_name: Table name
            
        Returns:
            List of placeholder field names
        """
        # Common field patterns
        fields = [
            f"K_{table_name.upper()}",  # Primary key
            "K_RECORD",
            "RECORD_ID",
            "CODE",
            "NAME",
            "DESCRIPTION",
            "STATUS",
            "TYPE",
            "DATE_CREATED",
            "DATE_MODIFIED",
            "CREATED_BY",
            "MODIFIED_BY",
            "ACTIVE",
            "DELETED"
        ]
        
        logging.info(f"Generated {len(fields)} placeholder fields for {table_name}")
        return fields
    
    def get_field_metadata(
        self,
        table_name: str,
        field_name: str
    ) -> Optional[Dict]:
        """
        Get metadata about a specific field.
        
        Args:
            table_name: Table name
            field_name: Field name
            
        Returns:
            Field metadata (type, description, constraints, etc.)
        """
        # TODO: Implement field-level metadata retrieval
        # For now, return basic info
        return {
            "field_name": field_name,
            "table_name": table_name,
            "data_type": "unknown",
            "nullable": True,
            "description": None
        }
