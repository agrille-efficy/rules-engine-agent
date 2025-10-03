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
    source_column: str
    target_column: str
    confidence: float = 0.0
    data_type_match: bool = False
    requires_transformation: bool = False
    transformation_note: Optional[str] = None 

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