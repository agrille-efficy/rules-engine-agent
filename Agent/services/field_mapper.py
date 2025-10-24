"""
Field Mapper Service - Maps CSV columns to database fields.
Uses multiple matching strategies: exact, semantic, fuzzy.
"""
import os
import logging
import re
from typing import List, Tuple, Dict, Optional
from difflib import SequenceMatcher

from ..models.file_analysis_model import FileAnalysisResult, ColumnMetadata
from ..models.rag_match_model import FieldMapping, FieldMappingResult, MappingValidationResult
from .llm_field_matcher import LLMFieldMatcherService

class FieldMapperService:
    """
    Service that maps CSV columns to database table fields.
    Uses intelligent matching algorithms with confidence scoring.
    """
    
    def __init__(self):
        self.exact_match_threshold = 1.0
        self.fuzzy_match_threshold = 0.69
        self.semantic_match_threshold = 0.6
        
        # Generic field terms that should be penalized
        self.generic_field_terms = [
            'metadata', 'memo', 'data', 'info', 'information',
            'ai', 'scoring', 'generic', 'misc', 'miscellaneous',
            'other', 'extra', 'additional', 'temp', 'temporary',
            'custom', 'field', 'column', 'value', 'blob'
        ]
        
        # Generic field penalty (reduces confidence by this factor)
        self.generic_field_penalty = 0.5  # 50% reduction
        
        # Type mismatch penalty for structured data → unstructured fields
        self.type_mismatch_penalty = 0.6  # 40% reduction
        
        # Maximum columns that can map to the same field before triggering refinement
        self.max_same_field_mappings = 2  # Changed from 3 to be more aggressive
        
        # Confidence threshold for removing suspicious mappings during refinement
        self.refinement_confidence_threshold = 0.72  # Catches 0.70 mappings

        self.llm_matcher = LLMFieldMatcherService()

        
    def map_fields(
        self,
        file_analysis: FileAnalysisResult,
        target_table_name: str,
        target_table_fields: List[str]
    ) -> FieldMappingResult:
        """
        Map CSV columns to database fields.
        
        Args:
            file_analysis: FileAnalysisResult from file analysis
            target_table_name: Name of selected database table
            target_table_fields: List of database field names
            
        Returns:
            FieldMappingResult with mappings and validation
        """
        logging.info(f"Mapping fields to table: {target_table_name}")
        
        mappings = []
        
        # Get source columns with English translations
        source_columns = file_analysis.columns
        
        for col in source_columns:
            # Try multiple matching strategies
            mapping = self._find_best_match(
                col,
                target_table_fields,
                file_analysis
            )
            
            if mapping:
                mappings.append(mapping)
                logging.info(f"Mapped: {col.name} → {mapping.target_column} "
                           f"({mapping.match_type}, {mapping.confidence_score:.2f})")
        
        # Post-mapping refinement
        mappings = self._refine_mappings(mappings, source_columns, target_table_fields)
        
        # Create result
        result = FieldMappingResult(
            mappings=mappings,
            source_table_name=file_analysis.structure.file_name,
            target_table_name=target_table_name,
            mapping_method="automatic"
        )
        
        # Validate mappings
        result.validation = self._validate_mappings(
            mappings,
            source_columns,
            target_table_fields
        )
        
        return result
    
    def _find_best_match(
        self,
        source_col: ColumnMetadata,
        target_fields: List[str],
        file_analysis: FileAnalysisResult
    ) -> Optional[FieldMapping]:
        """
        Find best matching target field for a source column.
        Tries: exact match → fuzzy match → LLM match → semantic match
        """
        # Use English name for matching (better results)
        col_name = source_col.english_name or source_col.name
        col_name_clean = self._normalize_name(col_name)
        
        best_match = None
        best_score = 0.0
        best_type = "unknown"
        
        for target_field in target_fields:
            target_clean = self._normalize_name(target_field)
            
            # Strategy 1: Exact match
            if col_name_clean == target_clean:
                return FieldMapping(
                    source_column=source_col.name,
                    source_column_english=col_name,
                    target_column=target_field,
                    confidence_score=1.0,
                    match_type="exact",
                    data_type_compatible=True,
                    source_data_type=source_col.data_type,
                    sample_source_values=source_col.sample_values[:3]
                )
            
            # Strategy 2: Fuzzy match (Levenshtein distance)
            fuzzy_score = self._fuzzy_similarity(col_name_clean, target_clean)
            if fuzzy_score > best_score and fuzzy_score >= self.fuzzy_match_threshold:
                best_score = fuzzy_score
                best_match = target_field
                best_type = "fuzzy"

            # Strategy 3: LLM-based match if fuzzy score is low
            if not hasattr(self, "llm_matcher") or self.llm_matcher is None:
                if best_match and best_score >= self.semantic_match_threshold:
                    return FieldMapping(
                        source_column=source_col.name,
                        source_columns_english=col_name,
                        target_column=best_match,
                        confidence_score=best_score,
                        match_type=best_type,
                        data_type_compatible=True,
                        source_data_type=source_col.data_type,
                        sample_source_values=source_col.sample_values[:3]
                    )
                return None
            
            llm_threshold = getattr(self, "llm_match_threshold", 0.65)
            table_context = f"Target table: {file_analysis.structure.file_name} ({len(target_fields)} fields)"

            try:
                llm_result = self.llm_matcher.find_llm_match(
                    source_col,
                    target_fields,
                    col_name,
                    table_context
                )
            except Exception as e:
                logging.debug(f"LLM matching failed for {source_col.name}: {e}")
                llm_result = None

            if llm_result:
                target_field = llm_result.get("target_field")
                confidence = llm_result.get("confidence", 0.0)
                if target_field in target_fields:
                    confidence *= self.generic_field_penalty
                if self._has_type_mismatch(source_col, target_field):
                    confidence *= self.type_mismatch_penalty
                if confidence > llm_threshold and confidence > best_score:
                    return FieldMapping(
                        source_column=source_col.name,
                        source_column_english=col_name,
                        target_column=target_field,
                        confidence_score=confidence,
                        match_type="llm",
                        data_type_compatible=True,
                        source_data_type=source_col.data_type,
                        sample_source_values=source_col.sample_values[:3]
                    )


            
            # Strategy 4: Semantic match (contains, prefix, suffix)
            semantic_score = self._semantic_similarity(col_name_clean, target_clean)
            
            # Apply generic field penalty
            if self._is_generic_field(target_field):
                semantic_score *= self.generic_field_penalty
                logging.debug(f"Applied generic field penalty to '{target_field}': {semantic_score:.2f}")
            
            # Apply type mismatch penalty
            if self._has_type_mismatch(source_col, target_field):
                semantic_score *= self.type_mismatch_penalty
                logging.debug(f"Applied type mismatch penalty for '{source_col.name}' → '{target_field}': {semantic_score:.2f}")
            
            if semantic_score > best_score and semantic_score >= self.semantic_match_threshold:
                best_score = semantic_score
                best_match = target_field
                best_type = "semantic"
        
        # Return best match if found
        if best_match and best_score >= self.semantic_match_threshold:
            return FieldMapping(
                source_column=source_col.name,
                source_column_english=col_name,
                target_column=best_match,
                confidence_score=best_score,
                match_type=best_type,
                data_type_compatible=True,
                source_data_type=source_col.data_type,
                sample_source_values=source_col.sample_values[:3]
            )
        
        return None
    
    def _is_generic_field(self, field_name: str) -> bool:
        """
        Check if a field name contains generic terms that suggest it's a catch-all field.
        """
        field_lower = field_name.lower()
        return any(term in field_lower for term in self.generic_field_terms)
    
    def _has_type_mismatch(self, source_col: ColumnMetadata, target_field: str) -> bool:
        """
        Detect if structured data (dates, numbers) is being mapped to unstructured fields (memo, blob).
        """
        # Check if target is an unstructured field
        target_lower = target_field.lower()
        is_unstructured_target = any(term in target_lower for term in ['memo', 'blob', 'metadata', 'data'])
        
        if not is_unstructured_target:
            return False
        
        # Check if source is structured data
        source_type_lower = source_col.data_type.lower()
        is_structured_source = any(term in source_type_lower for term in ['date', 'number', 'integer', 'decimal', 'numeric'])
        
        return is_structured_source
    
    def _refine_mappings(
        self,
        mappings: List[FieldMapping],
        source_columns: List[ColumnMetadata],
        target_fields: List[str]
    ) -> List[FieldMapping]:
        """
        Post-processing refinement to detect and fix suspicious mapping patterns.
        Identifies when too many columns map to the same field with identical scores.
        """
        logging.info(f"🔍 Starting refinement analysis on {len(mappings)} mappings...")
        
        # Count mappings per target field with their confidence scores
        target_field_analysis = {}
        for mapping in mappings:
            if mapping.target_column not in target_field_analysis:
                target_field_analysis[mapping.target_column] = []
            target_field_analysis[mapping.target_column].append(mapping)
        
        # Log the distribution
        logging.info(f"📊 Mappings distribution across {len(target_field_analysis)} fields:")
        for field, field_mappings in target_field_analysis.items():
            logging.info(f"   • {field}: {len(field_mappings)} mappings")
        
        # Detect overloaded fields - TWO criteria:
        # 1. Generic fields with > max_same_field_mappings
        # 2. ANY field with > max_same_field_mappings that has identical confidence scores (pattern issue)
        overloaded_fields = {}
        
        for field, field_mappings in target_field_analysis.items():
            count = len(field_mappings)
            
            if count <= self.max_same_field_mappings:
                continue
            
            is_generic = self._is_generic_field(field)
            
            # Check if all mappings have the same confidence (indicates algorithmic issue)
            confidence_scores = [m.confidence_score for m in field_mappings]
            has_identical_scores = len(set(confidence_scores)) == 1
            
            if is_generic:
                overloaded_fields[field] = {
                    'count': count,
                    'reason': 'generic_field',
                    'mappings': field_mappings
                }
                logging.warning(f"⚠️  Detected {count} mappings to GENERIC field '{field}'")
            elif has_identical_scores and count > self.max_same_field_mappings:
                overloaded_fields[field] = {
                    'count': count,
                    'reason': 'identical_scores',
                    'mappings': field_mappings
                }
                logging.warning(f"⚠️  Detected {count} mappings to '{field}' with IDENTICAL confidence ({confidence_scores[0]:.2f}) - likely algorithmic issue")
        
        if not overloaded_fields:
            logging.info("✅ No overloaded fields detected - refinement not needed")
            return mappings
        
        # Remove low-confidence mappings to overloaded fields
        refined_mappings = []
        removed_count = 0
        
        for mapping in mappings:
            if mapping.target_column in overloaded_fields:
                reason = overloaded_fields[mapping.target_column]['reason']
                
                # Remove if confidence is below threshold
                if mapping.confidence_score < self.refinement_confidence_threshold:
                    logging.info(f"❌ Removed mapping: {mapping.source_column} → {mapping.target_column} "
                               f"(score: {mapping.confidence_score:.2f}, reason: {reason})")
                    removed_count += 1
                else:
                    # Keep high-confidence mappings even to overloaded fields
                    logging.info(f"✅ Kept high-confidence mapping: {mapping.source_column} → {mapping.target_column} "
                               f"(score: {mapping.confidence_score:.2f})")
                    refined_mappings.append(mapping)
            else:
                refined_mappings.append(mapping)
        
        if removed_count > 0:
            logging.warning(f"🔧 Refinement: Removed {removed_count} suspicious mappings")
            logging.info(f"📉 Mappings reduced from {len(mappings)} to {len(refined_mappings)}")
        
        return refined_mappings
    
    def _normalize_name(self, name: str) -> str:
        """Normalize field name for comparison."""
        # Remove special chars, lowercase, remove spaces
        normalized = re.sub(r'[^a-z0-9]', '', name.lower())
        return normalized
    
    def _fuzzy_similarity(self, str1: str, str2: str) -> float:
        """Calculate fuzzy string similarity using SequenceMatcher."""
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _semantic_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate semantic similarity with more nuanced scoring.
        Checks for: exact substring matches, common prefixes/suffixes, word overlap.
        Returns scores between 0.0 and 1.0 with better granularity.
        """
        # Exact match (shouldn't reach here, but just in case)
        if str1 == str2:
            return 1.0
        
        # Calculate match scores
        scores = []
        
        # 1. Full substring match (one string contains the other)
        if str1 in str2 or str2 in str1:
            shorter = min(len(str1), len(str2))
            longer = max(len(str1), len(str2))
            # Score based on length ratio (longer matches = better)
            substring_score = 0.65 + (0.15 * (shorter / longer))
            scores.append(substring_score)
        
        # 2. Common prefix scoring (with length consideration)
        common_prefix_len = len(os.path.commonprefix([str1, str2]))
        if common_prefix_len >= 3:  # Minimum 3 chars
            # Score increases with prefix length
            prefix_score = min(0.80, 0.50 + (common_prefix_len / max(len(str1), len(str2)) * 0.30))
            scores.append(prefix_score)
        
        # 3. Word overlap (split by camelCase/snake_case)
        words1 = set(self._split_words(str1))
        words2 = set(self._split_words(str2))
        
        if words1 and words2:
            common_words = words1 & words2
            
            if common_words:
                # Calculate Jaccard similarity
                jaccard = len(common_words) / len(words1 | words2)
                
                # Check for important word matches (not just "oppo" prefix)
                important_matches = any(len(w) >= 4 for w in common_words)
                
                if important_matches:
                    # Better score for meaningful word overlaps
                    word_score = 0.55 + (jaccard * 0.25)
                else:
                    # Lower score for only short word matches (like "oppo")
                    word_score = 0.40 + (jaccard * 0.20)
                
                scores.append(word_score)
        
        # 4. Fuzzy similarity as fallback
        fuzzy_score = self._fuzzy_similarity(str1, str2)
        if fuzzy_score >= 0.5:  # Only consider if somewhat similar
            scores.append(fuzzy_score * 0.75)  # Penalize pure fuzzy matches
        
        # Return best score (or 0 if no matches)
        return max(scores) if scores else 0.0
    
    def _split_words(self, text: str) -> List[str]:
        """Split camelCase or snake_case into words."""
        # Split on underscores
        parts = text.split('_')
        words = []
        
        for part in parts:
            # Split camelCase
            words.extend(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', part))
        
        return [w.lower() for w in words if w]
    
    def _validate_mappings(
        self,
        mappings: List[FieldMapping],
        source_columns: List[ColumnMetadata],
        target_fields: List[str]
    ) -> MappingValidationResult:
        """
        Validate the quality of mappings.
        Checks coverage, confidence, unmapped fields.
        """
        total_source = len(source_columns)
        mapped_count = len(mappings)
        coverage_percent = (mapped_count / total_source * 100) if total_source > 0 else 0
        
        # Find unmapped columns
        mapped_source = {m.source_column for m in mappings}
        unmapped_source = [col.name for col in source_columns if col.name not in mapped_source]
        
        mapped_targets = {m.target_column for m in mappings}
        unmapped_targets = [f for f in target_fields if f not in mapped_targets]
        
        # Calculate average confidence
        avg_confidence = sum(m.confidence_score for m in mappings) / len(mappings) if mappings else 0
        
        # Determine confidence level
        if avg_confidence >= 0.8 and coverage_percent >= 80:
            confidence_level = "high"
        elif avg_confidence >= 0.6 and coverage_percent >= 60:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Collect issues and warnings
        issues = []
        warnings = []
        
        if coverage_percent < 50:
            issues.append(f"Low mapping coverage: {coverage_percent:.1f}%")
        
        if avg_confidence < 0.6:
            issues.append(f"Low average confidence: {avg_confidence:.2f}")
        
        if unmapped_source:
            warnings.append(f"{len(unmapped_source)} source columns unmapped: {', '.join(unmapped_source[:5])}")
        
        low_confidence_mappings = [m for m in mappings if m.confidence_score < 0.7]
        if low_confidence_mappings:
            warnings.append(f"{len(low_confidence_mappings)} mappings have low confidence")
        
        # Determine if valid
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
