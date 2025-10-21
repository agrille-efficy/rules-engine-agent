"""
Multi-Table Field Mapper Service - Maps CSV columns to multiple database tables.
Supports one-to-many file-to-tables mapping scenarios.
"""
import logging
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

from ..models.file_analysis_model import FileAnalysisResult, ColumnMetadata
from ..models.rag_match_model import FieldMapping, FieldMappingResult, MappingValidationResult
from .field_mapper import FieldMapperService
from .database_schema import DatabaseSchemaService


@dataclass
class TableFieldMapping:
    """Field mappings for a specific table."""
    table_name: str
    table_type: str  # Entity or Relation
    mappings: List[FieldMapping]
    validation: MappingValidationResult
    confidence_score: float
    insertion_order: int = 0  # Order for multi-table inserts (parent tables first)


@dataclass
class MultiTableMappingResult:
    """Result of multi-table field mapping."""
    source_file: str
    total_source_columns: int
    table_mappings: List[TableFieldMapping]
    overall_coverage: float
    overall_confidence: str
    unmapped_columns: List[str]
    is_valid: bool
    requires_review: bool


class MultiTableFieldMapper:
    """
    Maps CSV columns to multiple database tables simultaneously.
    Handles complex scenarios where data should be split across multiple tables.
    """
    
    def __init__(self):
        self.single_mapper = FieldMapperService()
        self.schema_service = DatabaseSchemaService()
        
        # Thresholds for multi-table mapping
        self.min_confidence_threshold = 0.5  # Lowered from 0.6 to capture more candidates
        self.min_columns_per_table = 1  # Minimum columns to consider a table relevant
        self.semantic_similarity_threshold = 0.75  # For column grouping
        
    def map_to_multiple_tables(
        self,
        file_analysis: FileAnalysisResult,
        candidate_tables: List[Dict],
        primary_table: str = None,
        max_tables: int = 5
    ) -> MultiTableMappingResult:
        """
        Map CSV columns to multiple database tables.
        
        Args:
            file_analysis: Result from file analysis
            candidate_tables: List of candidate tables from RAG (with scores)
            primary_table: The selected primary table to prioritize for mapping
            max_tables: Maximum number of tables to map to
            
        Returns:
            MultiTableMappingResult with mappings per table
        """
        logging.info(f"Starting multi-table mapping for {file_analysis.structure.file_name}")
        logging.info(f"Evaluating {len(candidate_tables)} candidate tables")
        if primary_table:
            logging.info(f"Primary table specified: {primary_table}")
        
        # Step 1: Get schemas for all candidate tables
        table_schemas = self._get_candidate_schemas(candidate_tables, max_tables)
        logging.info(f"Retrieved schemas for {len(table_schemas)} tables")
        
        # Step 2: Semantically group CSV columns by domain
        column_groups = self._group_columns_semantically(file_analysis)
        logging.info(f"Grouped columns into {len(column_groups)} semantic groups")
        
        # Step 3: Map each column to ALL potential tables (not just best)
        column_to_table_mappings = self._map_columns_to_all_tables(
            file_analysis,
            table_schemas
        )
        
        # Step 4: Smart assignment - distribute columns across tables
        table_assignments = self._smart_multi_table_assignment(
            column_to_table_mappings,
            column_groups,
            table_schemas,
            candidate_tables,
            primary_table  # Pass the primary table
        )
        
        # Step 5: Validate and filter tables
        valid_table_mappings = self._filter_and_validate_table_mappings(
            table_assignments,
            file_analysis,
            candidate_tables  # Pass candidate_tables for metadata lookup
        )
        
        # Step 6: Determine insertion order (parent tables first)
        valid_table_mappings = self._determine_insertion_order(
            valid_table_mappings,
            table_schemas
        )
        
        # Step 7: Calculate overall statistics
        result = self._create_final_result(
            file_analysis,
            valid_table_mappings
        )
        
        # Log summary
        self._log_summary(result)
        
        return result
    
    def _get_candidate_schemas(
        self,
        candidate_tables: List[Dict],
        max_tables: int
    ) -> Dict:
        """
        Get schemas for candidate tables.
        Prioritizes a balanced mix of Entity and Relation tables.
        Intelligently selects relationship tables based on relevance.
        """
        table_schemas = {}
        
        # Separate entity and relation tables
        entity_tables = []
        relation_tables = []
        
        for table_info in candidate_tables:
            table_kind = table_info.get('table_kind', 'Entity')
            if table_kind == 'Entity':
                entity_tables.append(table_info)
            else:
                relation_tables.append(table_info)
        
        # Fetch schemas with balanced approach:
        # - At least 3 entity tables (primary + alternates)
        # - Prioritize relationship tables that match common patterns
        tables_to_fetch = []
        
        # Add top entity tables (at least 3, or all if fewer)
        num_entities = min(3, len(entity_tables))
        tables_to_fetch.extend(entity_tables[:num_entities])
        
        # Fill remaining slots with relationship tables (SMARTLY)
        remaining_slots = max_tables - num_entities
        if remaining_slots > 0 and relation_tables:
            # Score relationship tables by their usefulness
            scored_relations = []
            for rel_table in relation_tables:
                table_name = rel_table.get('table_name', '')
                table_name_lower = table_name.lower()  # Use lowercase for pattern matching
                score = rel_table.get('composite_score', 0.0)
                
                # Boost score for commonly needed relationship patterns
                boost = 0.0
                if '_comp' in table_name_lower:  # Company relationships
                    boost = 0.5  # Increased from 0.3 to prioritize company tables
                elif '_user' in table_name_lower:  # User relationships
                    boost = 0.5  # Increased from 0.3 to prioritize user tables
                elif '_cont' in table_name_lower:  # Contact relationships
                    boost = 0.35  # Increased from 0.2
                elif 'intr_' in table_name_lower or '_intr' in table_name_lower:  # Interest/intervention
                    boost = 0.35  # Increased from 0.2
                elif '_prod' in table_name_lower:  # Product relationships
                    boost = 0.25  # Increased from 0.15
                elif '_proj' in table_name_lower:  # Project relationships
                    boost = 0.25  # Increased from 0.15
                
                adjusted_score = score + boost
                scored_relations.append((adjusted_score, rel_table))
            
            # Sort by adjusted score and take top ones
            scored_relations.sort(key=lambda x: x[0], reverse=True)
            top_relations = [rel for _, rel in scored_relations[:remaining_slots]]
            tables_to_fetch.extend(top_relations)
        
        # If we still have room and more entities, add them
        if len(tables_to_fetch) < max_tables and len(entity_tables) > num_entities:
            remaining = max_tables - len(tables_to_fetch)
            tables_to_fetch.extend(entity_tables[num_entities:num_entities + remaining])
        
        entity_count = len([t for t in tables_to_fetch if t.get('table_kind') == 'Entity'])
        relation_count = len([t for t in tables_to_fetch if t.get('table_kind') == 'Relation'])
        
        logging.info(f"Fetching schemas: {entity_count} Entity, {relation_count} Relation tables")
        
        # Log which relationship tables we're prioritizing
        rel_tables_names = [t.get('table_name') for t in tables_to_fetch if t.get('table_kind') == 'Relation']
        if rel_tables_names:
            logging.info(f"Prioritized relationship tables: {', '.join(rel_tables_names)}")
        
        # Fetch schemas
        for table_info in tables_to_fetch:
            table_name = table_info.get('table_name')
            if not table_name:
                continue
                
            fields = self.schema_service.get_table_fields(table_name)
            if fields:
                table_schemas[table_name] = {
                    'fields': fields,
                    'table_type': table_info.get('table_kind', 'Entity'),
                    'rag_score': table_info.get('composite_score', 0.0)
                }
                logging.info(f"  ✓ {table_name} ({table_info.get('table_kind', 'Entity')}): {len(fields)} fields")
        
        return table_schemas
    
    def _group_columns_semantically(
        self,
        file_analysis: FileAnalysisResult
    ) -> Dict[str, List[str]]:
        """
        Group CSV columns by semantic domain.
        Enhanced to better detect foreign keys and relationship fields.
        
        Returns:
            Dict mapping domain -> [column_names]
        """
        groups = {
            'entity_core': [],           # Core entity fields (name, status, type, nature)
            'entity_metadata': [],       # Entity metadata (dates, flags, archived)
            'entity_financial': [],      # Entity financial data (budget, amount, revenue)
            'foreign_keys': [],          # Foreign key fields (compUniqueID, userUniqueID, etc.)
            'user_references': [],       # User-related fields (responsible, creator, owner)
            'company_references': [],    # Company-related fields (company name, etc.)
            'contact_references': [],    # Contact-related fields
            'relationship_data': [],     # Data that belongs in relationship tables (vat, specific IDs)
            'other': []
        }
        
        for column in file_analysis.columns:
            col_name = column.name
            col_lower = (column.english_name or column.name).lower()
            
            # Priority 1: Detect foreign keys (highest priority)
            # Pattern: ends with UniqueID, _id, or contains ID patterns
            if (col_lower.endswith('uniqueid') or 
                col_lower.endswith('_id') or 
                (col_lower != 'id' and '_id' in col_lower)):
                groups['foreign_keys'].append(col_name)
                continue
            
            # Priority 2: Detect user/person references
            if any(pattern in col_lower for pattern in ['responsible', 'creator', 'owner', 'personne', 'charge', 'assignee']):
                groups['user_references'].append(col_name)
                continue
            
            # Priority 3: Detect company references (but not if already FK)
            if any(pattern in col_lower for pattern in ['company', 'comp', 'compte', 'société', 'organization']):
                # Check if it's about company name vs company ID
                if 'nom' in col_lower or 'name' in col_lower:
                    groups['company_references'].append(col_name)
                else:
                    groups['company_references'].append(col_name)
                continue
            
            # Priority 4: Detect contact references
            if any(pattern in col_lower for pattern in ['contact', 'cont']) and 'id' not in col_lower:
                groups['contact_references'].append(col_name)
                continue
            
            # Priority 5: Detect relationship-specific data
            if any(pattern in col_lower for pattern in ['intr_', 'vat', 'intervention', 'interest']):
                groups['relationship_data'].append(col_name)
                continue
            
            # Priority 6: Core entity fields
            if any(pattern in col_lower for pattern in ['oppo', 'opportunity', 'deal', 'affaire']) and \
               any(pattern in col_lower for pattern in ['name', 'status', 'type', 'nature', 'reason', 'nom', 'statut']):
                groups['entity_core'].append(col_name)
                continue
            
            # Priority 7: Entity metadata
            if any(pattern in col_lower for pattern in ['date', 'time', 'created', 'updated', 'sys', 'archived', 'banner']):
                groups['entity_metadata'].append(col_name)
                continue
            
            # Priority 8: Financial data
            if any(pattern in col_lower for pattern in ['amount', 'price', 'cost', 'revenue', 'budget', 'montant', 'loyer', 'capex', 'vente']):
                groups['entity_financial'].append(col_name)
                continue
            
            # Priority 9: Opportunity-specific fields (custom fields)
            if col_lower.startswith('oppo'):
                groups['entity_core'].append(col_name)
                continue
            
            # Default: other
            groups['other'].append(col_name)
        
        # Remove empty groups
        filtered_groups = {k: v for k, v in groups.items() if v}
        
        # Log the grouping results
        logging.info(f"Semantic grouping results:")
        for group_name, columns in filtered_groups.items():
            logging.info(f"  {group_name}: {len(columns)} columns")
        
        return filtered_groups
    
    def _map_columns_to_all_tables(
        self,
        file_analysis: FileAnalysisResult,
        table_schemas: Dict
    ) -> Dict[str, List[Dict]]:
        """
        Map each column to ALL possible tables with confidence scores.
        
        Returns:
            Dict mapping column_name -> [{table, field, confidence, mapping}, ...]
        """
        column_mappings = {}
        
        for column in file_analysis.columns:
            col_name = column.name
            column_mappings[col_name] = []
            
            # Try mapping to each table
            for table_name, schema_info in table_schemas.items():
                target_fields = schema_info['fields']
                
                # Find best match in this table
                best_match = self.single_mapper._find_best_match(
                    column,
                    target_fields,
                    file_analysis
                )
                
                if best_match and best_match.confidence_score >= self.min_confidence_threshold:
                    column_mappings[col_name].append({
                        'table': table_name,
                        'table_type': schema_info['table_type'],
                        'mapping': best_match,
                        'rag_score': schema_info['rag_score']
                    })
            
            # Sort by confidence (highest first)
            column_mappings[col_name].sort(
                key=lambda x: x['mapping'].confidence_score,
                reverse=True
            )
        
        return column_mappings
    
    def _smart_multi_table_assignment(
        self,
        column_to_table_mappings: Dict[str, List[Dict]],
        column_groups: Dict[str, List[str]],
        table_schemas: Dict,
        candidate_tables: List[Dict],
        primary_table: str = None
    ) -> Dict[str, List[FieldMapping]]:
        """
        Intelligently assign columns to multiple tables.
        
        Strategy:
        1. If primary_table is specified, prioritize mapping to it first (but exclude FKs)
        2. Explicitly map foreign keys and relationship fields to relationship tables
        3. Group remaining columns by semantic domain
        4. Find best table for each domain group
        5. Assign columns to tables based on group affinity and table type
        
        Returns:
            Dict mapping table_name -> [FieldMapping, ...]
        """
        table_assignments = {table: [] for table in table_schemas.keys()}
        assigned_columns = set()
        
        # Create lookup for table metadata from candidate_tables
        table_metadata_lookup = {
            t['table_name']: t for t in candidate_tables
        }
        
        # Step 1: Find table affinities for each semantic group
        group_to_table_affinity = self._calculate_group_table_affinity(
            column_groups,
            column_to_table_mappings,
            table_schemas
        )
        
        # Step 2: Track which tables have been used
        table_usage = {table: 0 for table in table_schemas.keys()}
        
        # Step 3: If primary_table is specified, map entity fields (but NOT foreign keys)
        if primary_table and primary_table in table_schemas:
            logging.info(f"=== PRIMARY TABLE MAPPING (SELECTIVE): {primary_table} ===")
            primary_table_meta = table_metadata_lookup.get(primary_table, {})
            primary_table_type = primary_table_meta.get('table_kind', 'Entity')
            logging.info(f"Primary table type: {primary_table_type}")
            
            # Collect columns to exclude from primary table mapping
            exclude_columns = set()
            exclude_columns.update(column_groups.get('foreign_keys', []))
            exclude_columns.update(column_groups.get('user_references', []))
            exclude_columns.update(column_groups.get('company_references', []))
            exclude_columns.update(column_groups.get('contact_references', []))
            exclude_columns.update(column_groups.get('relationship_data', []))
            
            if exclude_columns:
                logging.info(f"Excluding {len(exclude_columns)} FK/relationship columns from primary table")
            
            primary_assigned = 0
            
            for col_name, candidates in column_to_table_mappings.items():
                # Skip foreign keys and relationship fields
                if col_name in exclude_columns:
                    continue
                
                # Find mapping for primary table
                primary_mapping = next(
                    (c['mapping'] for c in candidates if c['table'] == primary_table),
                    None
                )
                
                if primary_mapping:
                    table_assignments[primary_table].append(primary_mapping)
                    assigned_columns.add(col_name)
                    primary_assigned += 1
            
            table_usage[primary_table] = primary_assigned
            logging.info(f"Assigned {primary_assigned} entity columns to primary table: {primary_table}")
        
        # Step 3.5: EXPLICIT RELATIONSHIP TABLE MAPPING for foreign keys
        logging.info(f"=== RELATIONSHIP TABLE MAPPING ===")
        
        # Get relationship tables
        relation_tables = [
            table for table in table_schemas.keys()
            if table_metadata_lookup.get(table, {}).get('table_kind') == 'Relation'
        ]
        
        logging.info(f"Found {len(relation_tables)} relationship tables to check")
        
        # EXPLICIT MAPPING: Map semantic groups to their corresponding relationship tables
        # This ensures user/company/contact references get properly mapped
        semantic_to_relation_mapping = {
            'user_references': ['_user', 'user_'],
            'company_references': ['_comp', 'comp_'],
            'contact_references': ['_cont', 'cont_']
        }
        
        for semantic_group, table_patterns in semantic_to_relation_mapping.items():
            group_columns = column_groups.get(semantic_group, [])
            if not group_columns:
                continue
            
            # Find the matching relationship table
            target_relation_table = None
            for rel_table in relation_tables:
                rel_table_lower = rel_table.lower()
                # Check if this relationship table matches the pattern and the primary entity
                if any(pattern in rel_table_lower for pattern in table_patterns):
                    if primary_table and primary_table[:4].lower() in rel_table_lower:
                        target_relation_table = rel_table
                        break
            
            if target_relation_table:
                # Map all columns from this group to the target relationship table
                mapped_count = 0
                for col_name in group_columns:
                    if col_name in assigned_columns:
                        continue
                    
                    # Find or create mapping for this column to the target table
                    candidates = column_to_table_mappings.get(col_name, [])
                    target_mapping = next(
                        (c['mapping'] for c in candidates if c['table'] == target_relation_table),
                        None
                    )
                    
                    # If no direct mapping exists (low confidence), create a relationship mapping
                    # with moderate confidence since we know semantically it belongs here
                    if not target_mapping:
                        # Try to find a reasonable field in the target table
                        from ..models.rag_match_model import FieldMapping
                        schema = table_schemas.get(target_relation_table, {})
                        target_fields = schema.get('fields', [])
                        
                        # Look for foreign key fields in the relationship table
                        # E.g., in Oppo_User, look for oppouserUserKey
                        suitable_field = None
                        for field in target_fields:
                            field_lower = field.lower()
                            # Match user FK patterns
                            if semantic_group == 'user_references' and ('userkey' in field_lower or 'user_key' in field_lower):
                                suitable_field = field
                                break
                            # Match company FK patterns
                            elif semantic_group == 'company_references' and ('companykey' in field_lower or 'company_key' in field_lower):
                                suitable_field = field
                                break
                            # Match contact FK patterns
                            elif semantic_group == 'contact_references' and ('contactkey' in field_lower or 'contact_key' in field_lower):
                                suitable_field = field
                                break
                        
                        if suitable_field:
                            target_mapping = FieldMapping(
                                source_column=col_name,
                                source_column_english=col_name,  # Use same name for English
                                target_column=suitable_field,
                                confidence_score=0.65,  # Moderate confidence for semantic mapping
                                match_type='semantic_group'
                            )
                    
                    if target_mapping:
                        table_assignments[target_relation_table].append(target_mapping)
                        assigned_columns.add(col_name)
                        table_usage[target_relation_table] += 1
                        mapped_count += 1
                        logging.info(f"  → Semantic '{col_name}' → '{target_relation_table}' (group: {semantic_group})")
                
                if mapped_count > 0:
                    logging.info(f"Mapped {mapped_count} columns from '{semantic_group}' group to '{target_relation_table}'")
        
        # EXPLICIT FK MAPPING: Handle foreign key columns (uniqueID patterns)
        # Map compUniqueID, userUniqueID etc. to their relationship tables
        fk_column_list = column_groups.get('foreign_keys', [])
        for fk_col_name in fk_column_list:
            if fk_col_name in assigned_columns:
                continue
            
            fk_col_lower = fk_col_name.lower()
            target_fk_table = None
            
            # Pattern matching for FK → Relationship table
            if 'compuniqueid' in fk_col_lower or 'comp_uniqueid' in fk_col_lower:
                # Find Oppo_Comp table
                target_fk_table = next((t for t in relation_tables if 'comp' in t.lower() and primary_table and primary_table[:4].lower() in t.lower()), None)
            elif 'useruniqueid' in fk_col_lower or 'user_uniqueid' in fk_col_lower:
                # Find Oppo_User table
                target_fk_table = next((t for t in relation_tables if 'user' in t.lower() and primary_table and primary_table[:4].lower() in t.lower()), None)
            elif 'contuniqueid' in fk_col_lower or 'cont_uniqueid' in fk_col_lower or 'contactuniqueid' in fk_col_lower:
                # Find Oppo_Cont table
                target_fk_table = next((t for t in relation_tables if 'cont' in t.lower() and primary_table and primary_table[:4].lower() in t.lower()), None)
            
            if target_fk_table:
                # Try to find existing mapping first
                candidates = column_to_table_mappings.get(fk_col_name, [])
                target_mapping = next(
                    (c['mapping'] for c in candidates if c['table'] == target_fk_table),
                    None
                )
                
                # If no mapping found, create one with moderate confidence
                if not target_mapping:
                    from ..models.rag_match_model import FieldMapping
                    schema = table_schemas.get(target_fk_table, {})
                    target_fields = schema.get('fields', [])
                    
                    # Look for CompanyKey or similar field
                    suitable_field = None
                    for field in target_fields:
                        field_lower = field.lower()
                        if 'companykey' in field_lower and 'comp' in fk_col_lower:
                            suitable_field = field
                            break
                        elif 'userkey' in field_lower and 'user' in fk_col_lower:
                            suitable_field = field
                            break
                        elif 'contactkey' in field_lower and 'cont' in fk_col_lower:
                            suitable_field = field
                            break
                    
                    if suitable_field:
                        target_mapping = FieldMapping(
                            source_column=fk_col_name,
                            source_column_english=fk_col_name,
                            target_column=suitable_field,
                            confidence_score=0.70,  # Good confidence for FK pattern match
                            match_type='foreign_key_pattern'
                        )
                
                if target_mapping:
                    table_assignments[target_fk_table].append(target_mapping)
                    assigned_columns.add(fk_col_name)
                    table_usage[target_fk_table] += 1
                    logging.info(f"  → FK '{fk_col_name}' → '{target_fk_table}' (pattern match)")
        
        # Continue with explicit FK/relationship column mapping
        fk_columns = []
        fk_columns.extend(column_groups.get('relationship_data', []))
        
        relationship_mapped = 0
        
        for col_name in fk_columns:
            if col_name in assigned_columns:
                continue
            
            candidates = column_to_table_mappings.get(col_name, [])
            if not candidates:
                continue
            
            # Try to find the best relationship table for this column
            best_relation_match = None
            best_relation_confidence = 0.0
            
            for candidate in candidates:
                table = candidate['table']
                mapping = candidate['mapping']
                confidence = mapping.confidence_score
                
                # Check if this is a relationship table
                if table in relation_tables:
                    # Use lower threshold for relationship tables (0.3 instead of 0.5)
                    if confidence >= 0.3 and confidence > best_relation_confidence:
                        # Additional check: does the table name match the column pattern?
                        is_good_match = self._is_good_relationship_match(
                            col_name, table, primary_table
                        )
                        
                        if is_good_match or confidence >= 0.5:
                            best_relation_match = (table, mapping)
                            best_relation_confidence = confidence
            
            # Assign to best relationship table
            if best_relation_match:
                rel_table, rel_mapping = best_relation_match
                table_assignments[rel_table].append(rel_mapping)
                assigned_columns.add(col_name)
                table_usage[rel_table] += 1
                relationship_mapped += 1
                logging.info(f"  → FK/Relation '{col_name}' → '{rel_table}' (confidence: {best_relation_confidence:.2f})")
        
        logging.info(f"Mapped {relationship_mapped} foreign key/relationship columns to relationship tables")
        
        # Step 4: Assign remaining columns by semantic groups
        for group_name, columns in column_groups.items():
            # Skip columns already assigned
            remaining_cols = [c for c in columns if c not in assigned_columns]
            if not remaining_cols:
                continue
            
            # Get best table(s) for this group
            best_tables = group_to_table_affinity.get(group_name, [])
            
            if not best_tables:
                logging.debug(f"No table affinity found for group '{group_name}'")
                continue
            
            # Choose best table for this group
            target_table = None
            best_affinity = 0.0
            
            for table_info in best_tables:
                table = table_info['table']
                affinity = table_info['affinity']
                
                # Skip if this is the primary table and we already mapped it
                if primary_table and table == primary_table:
                    continue
                
                # Get table type from metadata
                table_meta = table_metadata_lookup.get(table, {})
                table_type = table_meta.get('table_kind', 'Entity')
                is_entity = table_type == 'Entity'
                is_unused = table_usage[table] == 0
                
                # Scoring system:
                # - Entity tables get +0.1 bonus
                # - Unused tables get +0.05 bonus
                adjusted_affinity = affinity
                if is_entity:
                    adjusted_affinity += 0.1
                if is_unused:
                    adjusted_affinity += 0.05
                
                # Accept table if it meets threshold
                if adjusted_affinity >= 0.3 and adjusted_affinity > best_affinity:
                    target_table = table
                    best_affinity = adjusted_affinity
            
            if not target_table:
                continue
            
            # Log with table type information
            target_meta = table_metadata_lookup.get(target_table, {})
            target_type = target_meta.get('table_kind', 'Entity')
            logging.info(f"Group '{group_name}' ({len(remaining_cols)} cols) → Table '{target_table}' ({target_type}, affinity: {best_affinity:.2f})")
            
            # Assign remaining columns from this group
            assigned_count = 0
            for col_name in remaining_cols:
                candidates = column_to_table_mappings.get(col_name, [])
                
                # Find mapping for the target table
                target_mapping = next(
                    (c['mapping'] for c in candidates if c['table'] == target_table),
                    None
                )
                
                # If no mapping for target table, use best available
                if not target_mapping and candidates:
                    for candidate in candidates:
                        if primary_table and candidate['table'] == primary_table:
                            continue
                        target_mapping = candidate['mapping']
                        target_table = candidate['table']
                        break
                
                if target_mapping:
                    table_assignments[target_table].append(target_mapping)
                    assigned_columns.add(col_name)
                    assigned_count += 1
            
            if assigned_count > 0:
                table_usage[target_table] += assigned_count
                logging.info(f"  → Assigned {assigned_count} columns to {target_table}")
        
        # Step 5: Assign any remaining unmapped columns to their best table
        remaining = 0
        for col_name, candidates in column_to_table_mappings.items():
            if col_name in assigned_columns or not candidates:
                continue
            
            # Assign to best matching table (with lower threshold for leftovers)
            for candidate in candidates:
                if candidate['mapping'].confidence_score >= 0.3:
                    table = candidate['table']
                    table_assignments[table].append(candidate['mapping'])
                    assigned_columns.add(col_name)
                    table_usage[table] += 1
                    remaining += 1
                    break
        
        if remaining > 0:
            logging.info(f"Assigned {remaining} remaining columns to their best-match tables")
        
        return table_assignments
    
    def _is_good_relationship_match(
        self,
        column_name: str,
        relation_table: str,
        primary_table: str
    ) -> bool:
        """
        Check if a column is a good semantic match for a relationship table.
        
        Examples:
        - compUniqueID should match Oppo_Comp
        - responsible should match Oppo_User
        - intr_vat should match Intr_Oppo
        """
        col_lower = column_name.lower()
        table_lower = relation_table.lower()
        
        # Pattern 1: Column contains entity prefix that's in the table name
        # Example: compUniqueID matches Oppo_Comp
        entity_patterns = {
            'comp': ['comp', 'company'],
            'user': ['user', 'responsible', 'creator', 'owner'],
            'cont': ['cont', 'contact'],
            'prod': ['prod', 'product'],
            'intr': ['intr', 'interest', 'intervention'],
            'proj': ['proj', 'project'],
            'lead': ['lead'],
            'camp': ['camp', 'campaign']
        }
        
        for entity_prefix, patterns in entity_patterns.items():
            # Check if column matches any pattern
            if any(pattern in col_lower for pattern in patterns):
                # Check if the relationship table contains this entity prefix
                if entity_prefix in table_lower:
                    return True
        
        # Pattern 2: Check if primary table is in the relation table name
        # (All relationship tables should connect to the primary entity)
        if primary_table and primary_table.lower() in table_lower:
            # Additional check: does the column suggest this relationship?
            if any(pattern in col_lower for pattern in ['comp', 'user', 'cont', 'prod', 'intr']):
                return True
        
        return False
    
    def _calculate_group_table_affinity(
        self,
        column_groups: Dict[str, List[str]],
        column_to_table_mappings: Dict[str, List[Dict]],
        table_schemas: Dict
    ) -> Dict[str, List[Dict]]:
        """
        Calculate affinity scores between semantic groups and tables.
        
        Returns:
            Dict mapping group_name -> [{table, affinity_score}, ...] (sorted by affinity)
        """
        group_affinities = {}
        
        for group_name, columns in column_groups.items():
            table_scores = {}
            
            # Calculate how well each table matches this group's columns
            for col_name in columns:
                candidates = column_to_table_mappings.get(col_name, [])
                
                for candidate in candidates:
                    table = candidate['table']
                    confidence = candidate['mapping'].confidence_score
                    rag_score = candidate['rag_score']
                    
                    # Combined score: confidence + RAG relevance
                    score = (confidence * 0.7) + (rag_score * 0.3)
                    
                    if table not in table_scores:
                        table_scores[table] = []
                    table_scores[table].append(score)
            
            # Calculate average affinity for each table
            table_affinities = []
            for table, scores in table_scores.items():
                avg_score = sum(scores) / len(scores) if scores else 0.0
                coverage = len(scores) / len(columns) if columns else 0.0
                
                # Affinity = average score × coverage
                affinity = avg_score * (0.5 + (coverage * 0.5))
                
                table_affinities.append({
                    'table': table,
                    'affinity': affinity,
                    'coverage': coverage,
                    'avg_confidence': avg_score
                })
            
            # Sort by affinity (highest first)
            table_affinities.sort(key=lambda x: x['affinity'], reverse=True)
            group_affinities[group_name] = table_affinities
        
        return group_affinities
    
    def _filter_and_validate_table_mappings(
        self,
        table_groups: Dict[str, List[FieldMapping]],
        file_analysis: FileAnalysisResult,
        candidate_tables: List[Dict]
    ) -> List[TableFieldMapping]:
        """
        Filter tables that have too few mappings and validate each table's mappings.
        """
        # Create lookup for table metadata
        table_metadata_lookup = {
            t['table_name']: t for t in candidate_tables
        }
        
        valid_mappings = []
        
        for table_name, mappings in table_groups.items():
            if len(mappings) < self.min_columns_per_table:
                logging.debug(f"Skipping {table_name}: only {len(mappings)} columns mapped")
                continue
            
            # Get table schema info
            schema = self.schema_service.get_table_fields(table_name)
            
            # Validate this table's mappings
            validation = self.single_mapper._validate_mappings(
                mappings=mappings,
                source_columns=file_analysis.columns,
                target_fields=schema
            )
            
            # Calculate table confidence
            avg_confidence = sum(m.confidence_score for m in mappings) / len(mappings) if mappings else 0.0
            
            # Get table type from metadata (not from name pattern)
            table_meta = table_metadata_lookup.get(table_name, {})
            table_type = table_meta.get('table_kind', 'Entity')
            
            table_mapping = TableFieldMapping(
                table_name=table_name,
                table_type=table_type,
                mappings=mappings,
                validation=validation,
                confidence_score=avg_confidence
            )
            
            valid_mappings.append(table_mapping)
        
        # Sort by number of mappings (descending)
        valid_mappings.sort(key=lambda x: len(x.mappings), reverse=True)
        
        return valid_mappings
    
    def _determine_insertion_order(
        self,
        table_mappings: List[TableFieldMapping],
        table_schemas: Dict
    ) -> List[TableFieldMapping]:
        """
        Determine insertion order based on table dependencies.
        Parent tables (referenced by foreign keys) should be inserted first.
        """
        # Simple heuristic: Entity tables before Relation tables
        for i, tm in enumerate(table_mappings):
            if tm.table_type == "Entity":
                tm.insertion_order = 1
            else:
                tm.insertion_order = 2
        
        # Sort by insertion order
        table_mappings.sort(key=lambda x: x.insertion_order)
        
        return table_mappings
    
    def _create_final_result(
        self,
        file_analysis: FileAnalysisResult,
        valid_table_mappings: List[TableFieldMapping]
    ) -> MultiTableMappingResult:
        """Create final result object."""
        total_mapped = sum(len(tm.mappings) for tm in valid_table_mappings)
        total_source = file_analysis.structure.total_columns
        overall_coverage = (total_mapped / total_source * 100) if total_source > 0 else 0.0
        
        # Get unmapped columns
        mapped_columns = set()
        for tm in valid_table_mappings:
            mapped_columns.update(m.source_column for m in tm.mappings)
        
        all_columns = {col.name for col in file_analysis.columns}
        unmapped = list(all_columns - mapped_columns)
        
        # Determine overall confidence
        avg_confidence = sum(tm.confidence_score for tm in valid_table_mappings) / len(valid_table_mappings) if valid_table_mappings else 0.0
        overall_confidence = self._calculate_confidence_level(avg_confidence)
        
        is_valid = overall_coverage >= 50.0 and avg_confidence >= 0.6
        requires_review = not is_valid or overall_coverage < 70.0
        
        return MultiTableMappingResult(
            source_file=file_analysis.structure.file_name,
            total_source_columns=total_source,
            table_mappings=valid_table_mappings,
            overall_coverage=overall_coverage,
            overall_confidence=overall_confidence,
            unmapped_columns=unmapped,
            is_valid=is_valid,
            requires_review=requires_review
        )
    
    def _log_summary(self, result: MultiTableMappingResult):
        """Log a detailed summary of the mapping results."""
        logging.info("=" * 80)
        logging.info("MULTI-TABLE MAPPING SUMMARY")
        logging.info("=" * 80)
        logging.info(f"Source: {result.source_file}")
        logging.info(f"Total columns: {result.total_source_columns}")
        logging.info(f"Tables mapped: {len(result.table_mappings)}")
        logging.info(f"Overall coverage: {result.overall_coverage:.1f}%")
        logging.info(f"Overall confidence: {result.overall_confidence}")
        
        for tm in result.table_mappings:
            logging.info(f"\n  {tm.table_name} ({tm.table_type}) [Order: {tm.insertion_order}]:")
            logging.info(f"     Columns: {len(tm.mappings)}")
            logging.info(f"     Coverage: {tm.validation.mapping_coverage_percent:.1f}%")
            logging.info(f"     Confidence: {tm.validation.confidence_level}")
        
        if result.unmapped_columns:
            logging.warning(f"\n  [WARNING] Unmapped columns: {len(result.unmapped_columns)}")
            logging.warning(f"     {', '.join(result.unmapped_columns[:5])}...")
        
        logging.info("=" * 80)
    
    def _calculate_confidence_level(self, score: float) -> str:
        """Calculate confidence level from score."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"
