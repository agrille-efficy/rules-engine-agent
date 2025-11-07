"""
Data models for RAG matching results.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field 
from datetime import datetime

@dataclass
class TableMatch:
    """Represents a potential table match from RAG."""
    table_name: str
    schema_name: Optional[str] = None
    similarity_score: float = 0.0
    confidence: str = "low"  # low, medium, high
    matching_columns: List[str] = None
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.matching_columns is None:
            self.matching_columns = []
        if self.metadata is None: 
            self.metadata = {}
    
@dataclass 
class FieldMapping:
    """Represents a mapping from file column to database column."""
    source_column: str              # Original CSV column name
    source_column_english: str      # Translated English name
    target_column: str              # Database field name
    confidence_score: float = 0.0   # 0.0-1.0 mapping confidence
    match_type: str = "unknown"     # exact, semantic, fuzzy, manual
    data_type_compatible: bool = True
    requires_transformation: bool = False
    transformation_type: Optional[str] = None  # date_format, trim, uppercase, etc.
    transformation_note: Optional[str] = None
    source_data_type: Optional[str] = None
    target_data_type: Optional[str] = None
    sample_source_values: List[Any] = field(default_factory=list)
    validation_notes: List[str] = field(default_factory=list)

@dataclass
class MappingValidationResult:
    """Result of field mapping validation."""
    is_valid: bool = True
    confidence_level: str = "low"  # low, medium, high
    total_mappings: int = 0
    mapped_count: int = 0
    unmapped_source_columns: List[str] = field(default_factory=list)
    unmapped_target_columns: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mapping_coverage_percent: float = 0.0
    requires_review: bool = False
    validation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class FieldMappingResult:
    """Complete result of field mapping process."""
    mappings: List[FieldMapping] = field(default_factory=list)
    validation: Optional[MappingValidationResult] = None
    source_table_name: Optional[str] = None  # CSV filename
    target_table_name: Optional[str] = None  # Selected database table
    mapping_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    mapping_method: str = "automatic"  # automatic, manual, hybrid
    
    def get_mapped_columns(self) -> List[str]:
        """Get list of successfully mapped source columns."""
        return [m.source_column for m in self.mappings]
    
    def get_high_confidence_mappings(self) -> List[FieldMapping]:
        """Get only high confidence mappings (>= 0.7)."""
        return [m for m in self.mappings if m.confidence_score >= 0.7]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_table": self.source_table_name,
            "target_table": self.target_table_name,
            "mapping_method": self.mapping_method,
            "mappings": [
                {
                    "source_column": m.source_column,
                    "source_column_english": m.source_column_english,
                    "target_column": m.target_column,
                    "confidence_score": m.confidence_score,
                    "match_type": m.match_type,
                    "data_type_compatible": m.data_type_compatible,
                    "requires_transformation": m.requires_transformation,
                    "transformation_type": m.transformation_type
                }
                for m in self.mappings
            ],
            "validation": {
                "is_valid": self.validation.is_valid,
                "confidence_level": self.validation.confidence_level,
                "mapped_count": self.validation.mapped_count,
                "total_mappings": self.validation.total_mappings,
                "coverage_percent": self.validation.mapping_coverage_percent,
                "issues": self.validation.issues,
                "warnings": self.validation.warnings
            } if self.validation else None,
            "timestamp": self.mapping_timestamp
        }

@dataclass
class TableMatchResult:
    """Complete result of RAG table matching."""
    matched_tables: List[TableMatch] = field(default_factory=list)
    primary_match: Optional[TableMatch] = None
    suggested_mappings: List[FieldMapping] = field(default_factory=list)
    search_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    search_query: Optional[str] = None
    total_candidates: int = 0

    def __post_init__(self):
        """Auto-set primary match as highest scoring."""
        if not self.primary_match and self.matched_tables:
            self.primary_match = max(
                self.matched_tables,
                key=lambda x: x.similarity_score
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to dictionary for serialization."""
        return {
            "matched_tables": [
                {
                    "table_name": m.table_name,
                    "schema_name": m.schema_name,
                    "similarity_score": m.similarity_score,
                    "confidence": m.confidence,
                    "matching_columns": m.matching_columns,
                    "reason": m.reason,
                    "metadata": m.metadata
                }
                for m in self.matched_tables
            ],
            "primary_match": {
                "table_name": self.primary_match.table_name,
                "similarity_score": self.primary_match.similarity_score,
                "confidence": self.primary_match.confidence
            } if self.primary_match else None,
            "total_candidates": self.total_candidates,
            "search_timestamp": self.search_timestamp,
            "search_query": self.search_query
        }

@dataclass
class TableFieldMapping:
    """Field mappings for a specific table in multi-table mapping."""
    table_name: str
    table_type: str  # Entity or Relation
    mappings: List[FieldMapping]
    validation: MappingValidationResult
    confidence: float
    insertion_order: int = 0


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
    requires_refinement: bool = False