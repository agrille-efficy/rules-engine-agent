"""
Mapper Service - Unified multi-table mapping using exact, fuzzy, semantic, and LLM batch calls.
Replaces llm_field_matcher, field_mapper, and multi_table_mapper.
Includes: semantic grouping, relationship detection, refinement, and judge.
"""
import logging
import json
import re
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

from .clients.openai_client import ResilientOpenAIClient
from ..config import get_settings
from ..models.file_analysis_model import FileAnalysisResult, ColumnMetadata
from ..models.rag_match_model import (
    FieldMapping, 
    MappingValidationResult,
    TableFieldMapping,
    MultiTableMappingResult
)

settings = get_settings()


class Mapper:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        # Use resilient client with retry logic and circuit breaker
        self.client = ResilientOpenAIClient(
            api_key=settings.openai_api_key,
            model=model,
            temperature=temperature,
            max_retries=3
        )
        
        # Matching thresholds
        self.exact_match_threshold = 1.0
        self.fuzzy_match_threshold = 0.69
        self.semantic_match_threshold = 0.6
        self.llm_match_threshold = 0.5
        
        # Generic field detection
        self.generic_field_terms = [
            'metadata', 'memo', 'data', 'info', 'information',
            'ai', 'scoring', 'generic', 'misc', 'miscellaneous',
            'other', 'extra', 'additional', 'temp', 'temporary',
            'custom', 'field', 'column', 'value', 'blob', 'archived'
        ]
        
        # Penalties
        self.generic_field_penalty = 0.5
        self.type_mismatch_penalty = 0.6
        
        # Refinement settings
        self.max_same_field_mappings = 2
        self.refinement_confidence_threshold = 0.72
        
        # Multi-table settings
        self.min_confidence_threshold = 0.5
        self.min_columns_per_table = 1

    def map_to_multiple_tables(
        self,
        file_analysis: FileAnalysisResult,
        candidate_tables: List[Dict],
        primary_table: str = None,
        max_tables: int = 5
    ) -> MultiTableMappingResult:
        """
        Map CSV columns to multiple database tables with intelligent assignment.
        
        Args:
            file_analysis: Result from file analysis
            candidate_tables: List of candidate tables from RAG
            primary_table: Optional primary table to prioritize
            max_tables: Maximum number of tables to consider
            
        Returns:
            MultiTableMappingResult with mappings per table
        """
        logging.info(f"Starting multi-table mapping for {file_analysis.structure.file_name}")
        logging.info(f"Evaluating {len(candidate_tables)} candidate tables (max: {max_tables})")
        if primary_table:
            logging.info(f"Primary table: {primary_table}")
        
        # Step 1: Prepare table schemas
        prioritized_tables = candidate_tables[:max_tables]
        table_schemas = self._prepare_table_schemas(prioritized_tables)
        
        # Step 2: Semantic grouping of columns
        column_groups = self._group_columns_semantically(file_analysis)
        
        # Step 3: Map columns to ALL tables (multi-strategy)
        column_to_table_mappings = self._map_columns_to_all_tables(
            file_analysis,
            table_schemas
        )
        
        # Step 4: Smart multi-table assignment
        table_assignments = self._smart_multi_table_assignment(
            column_to_table_mappings,
            column_groups,
            table_schemas,
            candidate_tables,
            primary_table
        )
        
        # Step 5: Judge - analyze if refinement is needed
        needs_refinement = self._judge_needs_refinement(table_assignments, file_analysis)
        
        # Step 6: Refine if needed
        if needs_refinement:
            logging.warning("Judge recommends refinement")
            table_assignments = self._refine_all_table_mappings(table_assignments, table_schemas)
        else:
            logging.info("Judge approved mappings - no refinement needed")
        
        # Step 7: Filter and validate
        valid_table_mappings = self._filter_and_validate_table_mappings(
            table_assignments,
            file_analysis,
            candidate_tables
        )
        
        # Step 8: Determine insertion order
        valid_table_mappings = self._determine_insertion_order(valid_table_mappings, table_schemas)
        
        # Step 9: Create final result
        result = self._create_final_result(file_analysis, valid_table_mappings, needs_refinement)
        
        # Step 10: Log summary
        self._log_summary(result)
        
        return result

    # ============================================================================
    # SEMANTIC GROUPING
    # ============================================================================
    
    def _group_columns_semantically(self, file_analysis: FileAnalysisResult) -> Dict[str, List[str]]:
        """
        Group CSV columns by semantic domain.
        Detects: entity core, foreign keys, user/company/contact references, metadata, financial data.
        """
        groups = {
            'entity_core': [],
            'entity_metadata': [],
            'entity_financial': [],
            'foreign_keys': [],
            'user_references': [],
            'company_references': [],
            'contact_references': [],
            'relationship_data': [],
            'other': []
        }
        
        for column in file_analysis.columns:
            col_name = column.name
            col_lower = (column.english_name or column.name).lower()
            
            # Priority 1: Foreign keys
            if (col_lower.endswith('uniqueid') or col_lower.endswith('_id') or 
                (col_lower != 'id' and '_id' in col_lower)):
                groups['foreign_keys'].append(col_name)
            
            # Priority 2: User references
            elif any(p in col_lower for p in ['responsible', 'creator', 'owner', 'personne', 'charge', 'assignee']):
                groups['user_references'].append(col_name)
            
            # Priority 3: Company references
            elif any(p in col_lower for p in ['company', 'comp', 'compte', 'société', 'organization']):
                groups['company_references'].append(col_name)
            
            # Priority 4: Contact references
            elif any(p in col_lower for p in ['contact', 'cont']) and 'id' not in col_lower:
                groups['contact_references'].append(col_name)
            
            # Priority 5: Relationship data
            elif any(p in col_lower for p in ['intr_', 'vat', 'intervention', 'interest']):
                groups['relationship_data'].append(col_name)
            
            # Priority 6: Core entity fields
            elif any(p in col_lower for p in ['name', 'status', 'type', 'nature', 'reason', 'nom', 'statut']):
                groups['entity_core'].append(col_name)
            
            # Priority 7: Metadata
            elif any(p in col_lower for p in ['date', 'time', 'created', 'updated', 'sys', 'archived', 'banner']):
                groups['entity_metadata'].append(col_name)
            
            # Priority 8: Financial
            elif any(p in col_lower for p in ['amount', 'price', 'cost', 'revenue', 'budget', 'montant', 'loyer', 'capex', 'vente']):
                groups['entity_financial'].append(col_name)
            
            else:
                groups['other'].append(col_name)
        
        filtered_groups = {k: v for k, v in groups.items() if v}
        logging.info(f"Semantic grouping: {', '.join([f'{k}({len(v)})' for k, v in filtered_groups.items()])}")
        return filtered_groups

    # ============================================================================
    # MULTI-STRATEGY MAPPING (Exact, Fuzzy, Semantic, LLM)
    # ============================================================================
    
    def _map_columns_to_all_tables(
        self,
        file_analysis: FileAnalysisResult,
        table_schemas: Dict
    ) -> Dict[str, List[Dict]]:
        """
        Map each column to all tables using multiple strategies.
        Returns: {column_name: [{table, mapping, confidence, reasoning}, ...]}
        """
        columns_mappings = {}
        source_columns = file_analysis.columns
        
        for column in source_columns:
            col_name = column.name
            columns_mappings[col_name] = []
            
            # Try exact + fuzzy + semantic for each table
            for table_name, schema_info in table_schemas.items():
                target_fields = schema_info['fields']
                
                # Strategy 1-3: Exact, Fuzzy, Semantic
                best_match = self._find_best_non_llm_match(column, target_fields)
                
                if best_match:
                    columns_mappings[col_name].append({
                        'table': table_name,
                        'table_type': schema_info['table_type'],
                        'mapping': FieldMapping(
                            source_column=column.name,
                            source_column_english=column.english_name or column.name,
                            target_column=best_match['target_field'],
                            confidence_score=best_match['confidence'],
                            match_type=best_match['match_type'],
                            data_type_compatible=True,
                            source_data_type=column.data_type,
                            sample_source_values=column.sample_values[:3]
                        ),
                        'confidence': best_match['confidence'],
                        'reasoning': best_match.get('reasoning', '')
                    })
        
        # Strategy 4: Batch LLM for unmapped or low-confidence columns
        unmapped_columns = []
        for col_name, mappings in columns_mappings.items():
            if not mappings or max([m['confidence'] for m in mappings]) < 0.7:
                col = next(c for c in source_columns if c.name == col_name)
                unmapped_columns.append(col)
        
        if unmapped_columns:
            logging.info(f"Using LLM for {len(unmapped_columns)} columns with low/no matches")
            llm_results = self._batch_llm_mapping(unmapped_columns, table_schemas, file_analysis)
            
            for col_name, llm_matches in llm_results.items():
                for match in llm_matches:
                    # Add or replace existing mapping if LLM confidence is higher
                    existing = columns_mappings.get(col_name, [])
                    existing.append({
                        'table': match['table'],
                        'table_type': match['table_type'],
                        'mapping': match['mapping'],
                        'confidence': match['confidence'],
                        'reasoning': match['reasoning']
                    })
                    columns_mappings[col_name] = existing
        
        # Sort by confidence
        for col_name in columns_mappings:
            columns_mappings[col_name].sort(key=lambda x: x['confidence'], reverse=True)
        
        return columns_mappings
    
    def _find_best_non_llm_match(
        self,
        source_col: ColumnMetadata,
        target_fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Find best match using exact, fuzzy, and semantic strategies."""
        col_name = source_col.english_name or source_col.name
        col_clean = self._normalize_name(col_name)
        
        best_match = None
        best_score = 0.0
        best_type = "unknown"
        
        for target_field in target_fields:
            target_clean = self._normalize_name(target_field)
            
            # Strategy 1: Exact match
            if col_clean == target_clean:
                return {
                    'target_field': target_field,
                    'confidence': 1.0,
                    'match_type': 'exact',
                    'reasoning': 'Exact name match'
                }
            
            # Strategy 2: Fuzzy match
            fuzzy_score = self._fuzzy_similarity(col_clean, target_clean)
            if fuzzy_score >= self.fuzzy_match_threshold and fuzzy_score > best_score:
                best_score = fuzzy_score
                best_match = target_field
                best_type = "fuzzy"
            
            # Strategy 3: Semantic match
            semantic_score = self._semantic_similarity(col_clean, target_clean)
            
            # Apply penalties
            if self._is_generic_field(target_field):
                semantic_score *= self.generic_field_penalty
            if self._has_type_mismatch(source_col, target_field):
                semantic_score *= self.type_mismatch_penalty
            
            if semantic_score >= self.semantic_match_threshold and semantic_score > best_score:
                best_score = semantic_score
                best_match = target_field
                best_type = "semantic"
        
        if best_match and best_score >= self.semantic_match_threshold:
            return {
                'target_field': best_match,
                'confidence': best_score,
                'match_type': best_type,
                'reasoning': f'{best_type} match with score {best_score:.2f}'
            }
        
        return None
    
    def _fuzzy_similarity(self, str1: str, str2: str) -> float:
        """Calculate fuzzy string similarity using Levenshtein distance."""
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _semantic_similarity(self, str1: str, str2: str) -> float:
        """Calculate semantic similarity (substring, prefix, word overlap)."""
        if str1 == str2:
            return 1.0
        
        scores = []
        
        # Substring match
        if str1 in str2 or str2 in str1:
            shorter = min(len(str1), len(str2))
            longer = max(len(str1), len(str2))
            scores.append(0.65 + (0.15 * (shorter / longer)))
        
        # Common prefix
        common_prefix_len = len(self._common_prefix(str1, str2))
        if common_prefix_len >= 3:
            scores.append(min(0.80, 0.50 + (common_prefix_len / max(len(str1), len(str2)) * 0.30)))
        
        # Word overlap
        words1 = set(self._split_words(str1))
        words2 = set(self._split_words(str2))
        if words1 and words2:
            common_words = words1 & words2
            if common_words:
                jaccard = len(common_words) / len(words1 | words2)
                important_matches = any(len(w) >= 4 for w in common_words)
                if important_matches:
                    scores.append(0.55 + (jaccard * 0.25))
                else:
                    scores.append(0.40 + (jaccard * 0.20))
        
        return max(scores) if scores else 0.0
    
    def _common_prefix(self, str1: str, str2: str) -> str:
        """Find common prefix of two strings."""
        prefix = []
        for c1, c2 in zip(str1, str2):
            if c1 == c2:
                prefix.append(c1)
            else:
                break
        return ''.join(prefix)
    
    def _split_words(self, text: str) -> List[str]:
        """Split camelCase or snake_case into words."""
        parts = text.split('_')
        words = []
        for part in parts:
            words.extend(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', part))
        return [w.lower() for w in words if w]

    # ============================================================================
    # BATCH LLM MAPPING
    # ============================================================================
    
    def _batch_llm_mapping(
        self,
        columns: List[ColumnMetadata],
        table_schemas: Dict,
        file_analysis: FileAnalysisResult
    ) -> Dict[str, List[Dict]]:
        """Batch LLM mapping for multiple columns across multiple tables."""
        results = {}
        
        for table_name, schema_info in table_schemas.items():
            target_fields = schema_info['fields']
            table_type = schema_info['table_type']
            
            batch_prompt = self._build_batch_prompt(
                columns,
                target_fields,
                table_name,
                schema_info
            )
            
            llm_response = self._batch_llm_call(batch_prompt)
            
            for match in llm_response.get('matches', []):
                source_col = match.get('source_column')
                target_field = match.get('target_field')
                confidence = match.get('confidence', 0.0)
                reasoning = match.get('reasoning', '')
                
                if target_field and target_field != 'NO_MATCH' and confidence >= self.llm_match_threshold:
                    col = next((c for c in columns if c.name == source_col), None)
                    if col:
                        if source_col not in results:
                            results[source_col] = []
                        
                        results[source_col].append({
                            'table': table_name,
                            'table_type': table_type,
                            'mapping': FieldMapping(
                                source_column=col.name,
                                source_column_english=col.english_name or col.name,
                                target_column=target_field,
                                confidence_score=confidence,
                                match_type='llm',
                                data_type_compatible=True,
                                source_data_type=col.data_type,
                                sample_source_values=col.sample_values[:3]
                            ),
                            'confidence': confidence,
                            'reasoning': reasoning
                        })
        
        return results

    # ============================================================================
    # SMART MULTI-TABLE ASSIGNMENT
    # ============================================================================
    
    def _smart_multi_table_assignment(
        self,
        column_to_table_mappings: Dict[str, List[Dict]],
        column_groups: Dict[str, List[str]],
        table_schemas: Dict,
        candidate_tables: List[Dict],
        primary_table: str = None
    ) -> Dict[str, List[FieldMapping]]:
        """
        Intelligently assign columns to tables based on semantic groups and priorities.
        """
        table_assignments = {table: [] for table in table_schemas.keys()}
        assigned_columns = set()
        table_metadata_lookup = {t['table_name']: t for t in candidate_tables}
        
        # Step 1: Primary table entity fields (exclude FKs/relationships)
        if primary_table and primary_table in table_schemas:
            logging.info(f"Mapping entity fields to primary table: {primary_table}")
            exclude_groups = {'foreign_keys', 'user_references', 'company_references', 
                            'contact_references', 'relationship_data'}
            
            for col_name, candidates in column_to_table_mappings.items():
                # Skip if in excluded groups
                if any(col_name in column_groups.get(g, []) for g in exclude_groups):
                    continue
                
                primary_mapping = next((c['mapping'] for c in candidates if c['table'] == primary_table), None)
                if primary_mapping:
                    table_assignments[primary_table].append(primary_mapping)
                    assigned_columns.add(col_name)
        
        # Step 2: Relationship table mapping (FKs and references)
        logging.info(f"Mapping relationship fields to relationship tables")
        relation_tables = [t for t in table_schemas.keys() 
                          if table_metadata_lookup.get(t, {}).get('table_kind') == 'Relation']
        
        for semantic_group in ['foreign_keys', 'user_references', 'company_references', 
                               'contact_references', 'relationship_data']:
            group_cols = column_groups.get(semantic_group, [])
            for col_name in group_cols:
                if col_name in assigned_columns:
                    continue
                
                candidates = column_to_table_mappings.get(col_name, [])
                # Prefer relationship tables
                best_rel_match = next(
                    (c for c in candidates if c['table'] in relation_tables and c['confidence'] >= 0.3),
                    None
                )
                
                if best_rel_match:
                    table_assignments[best_rel_match['table']].append(best_rel_match['mapping'])
                    assigned_columns.add(col_name)
        
        # Step 3: Remaining columns to best matching table
        for col_name, candidates in column_to_table_mappings.items():
            if col_name in assigned_columns or not candidates:
                continue
            
            best_candidate = candidates[0]  # Already sorted by confidence
            if best_candidate['confidence'] >= self.min_confidence_threshold:
                table_assignments[best_candidate['table']].append(best_candidate['mapping'])
                assigned_columns.add(col_name)
        
        return table_assignments

    # ============================================================================
    # JUDGE - Decides if refinement is needed
    # ============================================================================
    
    def _judge_needs_refinement(
        self,
        table_assignments: Dict[str, List[FieldMapping]],
        file_analysis: FileAnalysisResult
    ) -> bool:
        """
        Judge analyzes mapping quality and decides if refinement is needed.
        
        Criteria for refinement:
        1. Overloaded fields (>2 columns to same field)
        2. Many identical confidence scores (algorithmic issue)
        3. Too many generic field mappings
        4. Low overall confidence
        """
        logging.info("Judge analyzing mapping quality...")
        
        reasons = []
        
        for table_name, mappings in table_assignments.items():
            if not mappings:
                continue
            
            # Check 1: Overloaded fields
            target_field_counts = {}
            for mapping in mappings:
                if mapping.target_column:
                    target_field_counts[mapping.target_column] = target_field_counts.get(mapping.target_column, 0) + 1
            
            overloaded = {f: c for f, c in target_field_counts.items() if c > self.max_same_field_mappings}
            if overloaded:
                reasons.append(f"Table {table_name}: {len(overloaded)} overloaded fields")
            
            # Check 2: Identical confidence scores
            confidences = [m.confidence_score for m in mappings if m.confidence_score > 0]
            if len(confidences) > 3 and len(set(confidences)) == 1:
                reasons.append(f"Table {table_name}: identical confidence scores detected")
            
            # Check 3: Generic field mappings
            generic_mappings = [m for m in mappings if m.target_column and self._is_generic_field(m.target_column)]
            if len(generic_mappings) > len(mappings) * 0.3:
                reasons.append(f"Table {table_name}: {len(generic_mappings)} generic field mappings")
            
            # Check 4: Low average confidence
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            if avg_conf < 0.65:
                reasons.append(f"Table {table_name}: low avg confidence ({avg_conf:.2f})")
        
        if reasons:
            logging.warning(f"Judge found {len(reasons)} issues:")
            for reason in reasons:
                logging.warning(f"   • {reason}")
            return True
        
        return False

    # ============================================================================
    # REFINER - Removes suspicious mappings
    # ============================================================================
    
    def _refine_all_table_mappings(
        self,
        table_assignments: Dict[str, List[FieldMapping]],
        table_schemas: Dict
    ) -> Dict[str, List[FieldMapping]]:
        """Refine mappings for all tables."""
        refined_assignments = {}
        
        for table_name, mappings in table_assignments.items():
            refined_mappings = self._refine_table_mappings(table_name, mappings)
            refined_assignments[table_name] = refined_mappings
        
        return refined_assignments
    
    def _refine_table_mappings(
        self,
        table_name: str,
        mappings: List[FieldMapping]
    ) -> List[FieldMapping]:
        """
        Refine mappings for a single table by detecting overloaded fields.
        Removes low-confidence mappings when too many columns map to same field.
        """
        logging.info(f"Refining mappings for table: {table_name} ({len(mappings)} mappings)")
        
        # Count mappings per target field
        target_field_analysis = {}
        for mapping in mappings:
            if mapping.target_column:
                if mapping.target_column not in target_field_analysis:
                    target_field_analysis[mapping.target_column] = []
                target_field_analysis[mapping.target_column].append(mapping)
        
        # Detect overloaded fields
        overloaded_fields = {}
        for field, field_mappings in target_field_analysis.items():
            count = len(field_mappings)
            if count <= self.max_same_field_mappings:
                continue
            
            is_generic = self._is_generic_field(field)
            confidence_scores = [m.confidence_score for m in field_mappings]
            has_identical_scores = len(set(confidence_scores)) == 1
            
            if is_generic or (has_identical_scores and count > self.max_same_field_mappings):
                reason = 'generic_field' if is_generic else 'identical_scores'
                overloaded_fields[field] = {'count': count, 'reason': reason, 'mappings': field_mappings}
                logging.warning(f"Overloaded: '{field}' ({count} mappings, reason: {reason})")
        
        if not overloaded_fields:
            logging.info("No overloaded fields detected")
            return mappings
        
        # Remove low-confidence mappings to overloaded fields
        refined_mappings = []
        removed_count = 0
        
        for mapping in mappings:
            if mapping.target_column in overloaded_fields:
                if mapping.confidence_score < self.refinement_confidence_threshold:
                    logging.info(f"Removed: {mapping.source_column} → {mapping.target_column} "
                               f"(score: {mapping.confidence_score:.2f})")
                    removed_count += 1
                else:
                    refined_mappings.append(mapping)
            else:
                refined_mappings.append(mapping)
        
        if removed_count > 0:
            logging.warning(f"Refinement: Removed {removed_count} suspicious mappings")
            logging.info(f"Mappings reduced: {len(mappings)} → {len(refined_mappings)}")
        
        return refined_mappings

    # ============================================================================
    # VALIDATION & FILTERING
    # ============================================================================
    
    def _filter_and_validate_table_mappings(
        self,
        table_assignments: Dict[str, List[FieldMapping]],
        file_analysis: FileAnalysisResult,
        candidate_tables: List[Dict]
    ) -> List[TableFieldMapping]:
        """Filter tables with too few mappings and validate."""
        table_metadata_lookup = {t['table_name']: t for t in candidate_tables}
        valid_mappings = []
        
        for table_name, mappings in table_assignments.items():
            if len(mappings) < self.min_columns_per_table:
                logging.debug(f"Skipping {table_name}: only {len(mappings)} columns mapped")
                continue
            
            # Validate - extract field names properly
            all_target_fields = []
            for t in candidate_tables:
                if t['table_name'] == table_name:
                    fields_raw = t.get('fields', [])
                    # Handle both list of strings and list of dicts
                    if fields_raw and isinstance(fields_raw[0], dict):
                        all_target_fields = [f.get('field_name', f.get('name', '')) for f in fields_raw if isinstance(f, dict)]
                    elif fields_raw and isinstance(fields_raw[0], str):
                        all_target_fields = fields_raw
                    break
            
            validation = self._validate_mappings(mappings, file_analysis.columns, all_target_fields)
            
            avg_confidence = sum(m.confidence_score for m in mappings) / len(mappings) if mappings else 0.0
            table_type = table_metadata_lookup.get(table_name, {}).get('table_kind', 'Entity')
            
            valid_mappings.append(TableFieldMapping(
                table_name=table_name,
                table_type=table_type,
                mappings=mappings,
                validation=validation,
                confidence=avg_confidence
            ))
        
        valid_mappings.sort(key=lambda x: len(x.mappings), reverse=True)
        return valid_mappings
    
    def _determine_insertion_order(
        self,
        table_mappings: List[TableFieldMapping],
        table_schemas: Dict
    ) -> List[TableFieldMapping]:
        """Determine insertion order (Entity tables first, then Relation tables)."""
        for i, tm in enumerate(table_mappings):
            tm.insertion_order = 1 if tm.table_type == "Entity" else 2
        
        table_mappings.sort(key=lambda x: x.insertion_order)
        return table_mappings

    # ============================================================================
    # RESULT CREATION & LOGGING
    # ============================================================================
    
    def _create_final_result(
        self,
        file_analysis: FileAnalysisResult,
        valid_table_mappings: List[TableFieldMapping],
        needs_refinement: bool
    ) -> MultiTableMappingResult:
        """Create final result object."""
        total_mapped = sum(len(tm.mappings) for tm in valid_table_mappings)
        total_source = file_analysis.structure.total_columns
        overall_coverage = (total_mapped / total_source * 100) if total_source > 0 else 0.0
        
        mapped_columns = set()
        for tm in valid_table_mappings:
            mapped_columns.update(m.source_column for m in tm.mappings if m.target_column)
        
        all_columns = {col.name for col in file_analysis.columns}
        unmapped = list(all_columns - mapped_columns)
        
        avg_confidence = sum(tm.confidence for tm in valid_table_mappings) / len(valid_table_mappings) if valid_table_mappings else 0.0
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
            requires_review=requires_review,
            requires_refinement=needs_refinement
        )
    
    def _calculate_confidence_level(self, score: float) -> str:
        """Calculate confidence level from score."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _log_summary(self, result: MultiTableMappingResult):
        """Log detailed summary of mapping results."""
        logging.info("=" * 80)
        logging.info("MULTI-TABLE MAPPING SUMMARY")
        logging.info("=" * 80)
        logging.info(f"Source: {result.source_file}")
        logging.info(f"Total columns: {result.total_source_columns}")
        logging.info(f"Tables mapped: {len(result.table_mappings)}")
        logging.info(f"Overall coverage: {result.overall_coverage:.1f}%")
        logging.info(f"Overall confidence: {result.overall_confidence}")
        logging.info(f"Refinement applied: {'Yes' if result.requires_refinement else 'No'}")
        logging.info(f"Requires review: {'Yes' if result.requires_review else 'No'}")
        
        for tm in result.table_mappings:
            logging.info(f"\n  {tm.table_name} ({tm.table_type}) [Order: {tm.insertion_order}]:")
            logging.info(f"     • Columns: {len(tm.mappings)}")
            logging.info(f"     • Coverage: {tm.validation.mapping_coverage_percent:.1f}%")
            logging.info(f"     • Confidence: {tm.validation.confidence_level}")
        
        if result.unmapped_columns:
            logging.warning(f"\n  Unmapped columns ({len(result.unmapped_columns)}):")
            logging.warning(f"     {', '.join(result.unmapped_columns[:10])}")
        
        logging.info("=" * 80)

    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _prepare_table_schemas(self, candidate_tables: List[Dict]) -> Dict:
        """Prepare table schemas from candidate tables."""
        table_schemas = {}
        for table in candidate_tables:
            table_name = table.get('table_name')
            if table_name:
                # Extract field names from metadata
                fields_raw = table.get('fields', []) or table.get('metadata', {}).get('table_fields', [])
                
                # Handle both list of strings and list of dicts
                if fields_raw and isinstance(fields_raw[0], dict):
                    fields = [f.get('field_name', f.get('name', '')) for f in fields_raw if isinstance(f, dict)]
                elif fields_raw and isinstance(fields_raw[0], str):
                    fields = fields_raw
                else:
                    fields = []
                
                table_schemas[table_name] = {
                    'fields': fields,
                    'table_type': table.get('table_kind', 'Entity'),
                    'rag_score': table.get('composite_score', 0.0)
                }
        return table_schemas

    def _normalize_name(self, name: str) -> str:
        """Normalize field name for comparison."""
        return re.sub(r'[^a-z0-9]', '', name.lower())
    
    def _is_generic_field(self, field_name: str) -> bool:
        """Check if field is generic/catch-all."""
        field_lower = field_name.lower()
        return any(term in field_lower for term in self.generic_field_terms)
    
    def _has_type_mismatch(self, source_col: ColumnMetadata, target_field: str) -> bool:
        """Detect structured data → unstructured field mismatch."""
        target_lower = target_field.lower()
        is_unstructured = any(term in target_lower for term in ['memo', 'blob', 'metadata', 'data'])
        if not is_unstructured:
            return False
        
        source_type_lower = source_col.data_type.lower()
        is_structured = any(term in source_type_lower for term in ['date', 'number', 'integer', 'decimal', 'numeric'])
        return is_structured

    def _build_batch_prompt(self, columns, target_fields, table_name, table_metadata):
        col_descriptions = []
        for col in columns:
            samples = getattr(col, 'sample_values', [])
            sample_text = ', '.join([f'"{v}"' for v in samples[:5]]) if samples else 'No samples'
            col_descriptions.append(
                f"- Name: {col.name}, English: {getattr(col, 'english_name', col.name)}, Type: {getattr(col, 'data_type', 'string')}, Samples: {sample_text}"
            )
        prompt = f"Match these source columns to the best target fields in table '{table_name}':\n"
        prompt += '\n'.join(col_descriptions)
        prompt += f"\nTarget fields: {', '.join(target_fields)}\nTable metadata: {table_metadata}\n"
        prompt += "Return JSON: {matches: [{source_column, target_field, confidence, reasoning}]}"
        return prompt

    def _batch_llm_call(self, prompt: str) -> Dict[str, Any]:
        """Call LLM using resilient client with retry logic and circuit breaker."""
        try:
            # Use resilient client's generate_completion method
            response_text = self.client.generate_completion(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response_text)
            return result if isinstance(result, dict) and "matches" in result else {"matches": []}
        except Exception as e:
            logging.error(f"Batch LLM call failed: {e}")
            return {"matches": []}

    def _get_system_prompt(self) -> str:
        return (
            "You are an expert data engineer specializing in database schema mapping and ETL processes.\n"
            "Your task is to match CSV column names to database table fields by analyzing:\n"
            "- Column names and their semantic meaning\n"
            "- Sample data values and patterns\n"
            "- Data types and formats\n"
            "- Business domain context\n"
            "- Common database naming conventions\n"
            "Guidelines:\n"
            "1. Consider the SEMANTIC MEANING, not just string similarity\n"
            "2. Avoid mapping structured data (dates, numbers) to generic memo/blob fields\n"
            "3. Penalize matches to overly generic fields (metadata, memo, misc, data, blob, etc.)\n"
            "4. Consider data type compatibility\n"
            "5. Use sample values to validate the match makes sense\n"
            "6. Be conservative - if uncertain, return NO_MATCH rather than guessing\n"
            "7. Provide clear reasoning for your decision\n"
            "8. Avoid at all costs mapping to or from fields with capital F in their name.\n"
            "Return JSON: {matches: [{source_column, target_field, confidence, reasoning}]}"
        )

    def _validate_mappings(self, mappings: List[FieldMapping], source_columns: List[ColumnMetadata], target_fields: List[str]) -> MappingValidationResult:
        """Validate mapping quality with coverage, confidence, and unmapped tracking."""
        total_source = len(source_columns)
        mapped_count = len([m for m in mappings if m.target_column])
        coverage_percent = (mapped_count / total_source * 100) if total_source > 0 else 0
        
        mapped_source = {m.source_column for m in mappings if m.target_column}
        unmapped_source = [col.name for col in source_columns if col.name not in mapped_source]
        
        mapped_targets = {m.target_column for m in mappings if m.target_column}
        unmapped_targets = [f for f in target_fields if f not in mapped_targets]
        
        avg_confidence = sum(m.confidence_score for m in mappings if m.target_column) / mapped_count if mapped_count else 0
        
        if avg_confidence >= 0.8 and coverage_percent >= 80:
            confidence_level = "high"
        elif avg_confidence >= 0.6 and coverage_percent >= 60:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        issues = []
        warnings = []
        
        if coverage_percent < 50:
            issues.append(f"Low mapping coverage: {coverage_percent:.1f}%")
        if avg_confidence < 0.6:
            issues.append(f"Low average confidence: {avg_confidence:.2f}")
        if unmapped_source:
            warnings.append(f"{len(unmapped_source)} source columns unmapped")
        
        low_confidence = [m for m in mappings if m.target_column and m.confidence_score < 0.7]
        if low_confidence:
            warnings.append(f"{len(low_confidence)} mappings have low confidence")
        
        is_valid = len(issues) == 0 and coverage_percent >= 50
        requires_review = confidence_level == "low" or len(issues) > 0
        
        return MappingValidationResult(
            is_valid=is_valid,
            confidence_level=confidence_level,
            total_mappings=total_source,
            mapped_count=mapped_count,
            unmapped_source_columns=unmapped_source,
            unmapped_target_columns=unmapped_targets,
            issues=issues,
            warnings=warnings,
            mapping_coverage_percent=coverage_percent,
            requires_review=requires_review
        )