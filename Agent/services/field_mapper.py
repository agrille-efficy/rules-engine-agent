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


class FieldMapperService:
    """
    Service that maps CSV columns to database table fields.
    Uses intelligent matching algorithms with confidence scoring.
    """
    
    def __init__(self):
        self.exact_match_threshold = 1.0
        self.fuzzy_match_threshold = 0.8
        self.semantic_match_threshold = 0.6
        
    def map_fields(
        self,
        file_analysis: FileAnalysisResult,
        target_table_name: str,
        target_table_fields: List[str]  # Will come from DICO API or metadata
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
        Tries: exact match → fuzzy match → semantic match
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
            
            # Strategy 3: Semantic match (contains, prefix, suffix)
            semantic_score = self._semantic_similarity(col_name_clean, target_clean)
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
                data_type_compatible=True,  # TODO: Add type checking
                source_data_type=source_col.data_type,
                sample_source_values=source_col.sample_values[:3]
            )
        
        return None
    
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
        Calculate semantic similarity.
        Checks for: substring matches, common prefixes/suffixes, word overlap.
        """
        score = 0.0
        
        # Full substring match
        if str1 in str2 or str2 in str1:
            score = 0.75
        
        # Common prefix (at least 4 chars)
        common_prefix_len = len(os.path.commonprefix([str1, str2]))
        if common_prefix_len >= 4:
            score = max(score, 0.7)
        
        # Word overlap (split by underscore/camelCase)
        words1 = self._split_words(str1)
        words2 = self._split_words(str2)
        common_words = set(words1) & set(words2)
        
        if common_words:
            overlap_ratio = len(common_words) / max(len(words1), len(words2))
            score = max(score, overlap_ratio * 0.8)
        
        return score
    
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
