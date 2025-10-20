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
        """Get schemas for candidate tables."""
        table_schemas = {}
        for table_info in candidate_tables[:max_tables]:
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
        return table_schemas
    
    def _group_columns_semantically(
        self,
        file_analysis: FileAnalysisResult
    ) -> Dict[str, List[str]]:
        """
        Group CSV columns by semantic domain.
        
        Returns:
            Dict mapping domain -> [column_names]
        """
        groups = {
            'identification': [],
            'company_info': [],
            'contact_info': [],
            'opportunity_info': [],
            'financial': [],
            'dates_times': [],
            'status_metadata': [],
            'relationships': [],
            'other': []
        }
        
        for column in file_analysis.columns:
            col_name = (column.english_name or column.name).lower()
            
            # Categorize by semantic patterns
            if any(pattern in col_name for pattern in ['id', 'key', 'uniqueid', 'k_']):
                groups['identification'].append(column.name)
            elif any(pattern in col_name for pattern in ['company', 'comp', 'organization', 'compte', 'société']):
                groups['company_info'].append(column.name)
            elif any(pattern in col_name for pattern in ['contact', 'person', 'personne', 'responsible', 'owner']):
                groups['contact_info'].append(column.name)
            elif any(pattern in col_name for pattern in ['oppo', 'opportunity', 'deal', 'affaire', 'nature']):
                groups['opportunity_info'].append(column.name)
            elif any(pattern in col_name for pattern in ['amount', 'price', 'cost', 'revenue', 'montant', 'vat']):
                groups['financial'].append(column.name)
            elif any(pattern in col_name for pattern in ['date', 'time', 'created', 'updated', 'closing']):
                groups['dates_times'].append(column.name)
            elif any(pattern in col_name for pattern in ['status', 'state', 'stage', 'reason', 'statut']):
                groups['status_metadata'].append(column.name)
            elif any(pattern in col_name for pattern in ['ref', 'link', 'relation']):
                groups['relationships'].append(column.name)
            else:
                groups['other'].append(column.name)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
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
        1. If primary_table is specified, prioritize mapping to it first
        2. Group columns by semantic domain
        3. Find best table for each domain group
        4. Assign columns to tables based on group affinity and table type
        5. Force distribution when appropriate (avoid putting everything in one table)
        
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
        
        # Step 3: If primary_table is specified, try to map as many columns as possible to it first
        if primary_table and primary_table in table_schemas:
            logging.info(f"=== PRIMARY TABLE PRIORITIZATION: {primary_table} ===")
            primary_table_meta = table_metadata_lookup.get(primary_table, {})
            primary_table_type = primary_table_meta.get('table_kind', 'Entity')
            logging.info(f"Primary table type: {primary_table_type}")
            
            primary_assigned = 0
            
            for col_name, candidates in column_to_table_mappings.items():
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
            logging.info(f"Assigned {primary_assigned} columns to primary table: {primary_table}")
        
        # Step 4: Assign remaining columns by semantic groups
        for group_name, columns in column_groups.items():
            # Skip columns already assigned to primary table
            remaining_cols = [c for c in columns if c not in assigned_columns]
            if not remaining_cols:
                continue
            
            # Get best table(s) for this group
            best_tables = group_to_table_affinity.get(group_name, [])
            
            if not best_tables:
                logging.debug(f"No table affinity found for group '{group_name}'")
                continue
            
            # Choose best table for this group (prefer unused tables for distribution)
            # Priority: Entity tables > Relation tables, unused > used, high affinity > low affinity
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
                
                # Accept table if:
                # 1. It has good adjusted affinity (>0.4), OR
                # 2. It's the best we've seen so far and meets minimum threshold (>0.3)
                if adjusted_affinity >= 0.4 and adjusted_affinity > best_affinity:
                    target_table = table
                    best_affinity = adjusted_affinity
                    break
                elif adjusted_affinity >= 0.3 and adjusted_affinity > best_affinity:
                    target_table = table
                    best_affinity = adjusted_affinity
            
            # Fallback to best table (skip primary if already used)
            if not target_table and best_tables:
                for table_info in best_tables:
                    if primary_table and table_info['table'] == primary_table:
                        continue
                    target_table = table_info['table']
                    best_affinity = table_info['affinity']
                    break
            
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
                
                # If no mapping for target table, use best available (excluding primary if already used)
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
            
            # Assign to best matching table
            best_candidate = candidates[0]
            table_assignments[best_candidate['table']].append(best_candidate['mapping'])
            assigned_columns.add(col_name)
            table_usage[best_candidate['table']] += 1
            remaining += 1
        
        if remaining > 0:
            logging.info(f"Assigned {remaining} remaining columns to their best-match tables")
        
        return table_assignments
    
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
