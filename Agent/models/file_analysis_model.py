"""
File Analysis data models.

These models represent the structured results from analyzing different file types
(CSV, Excel, JSON, PDF, images, text) for database ingestion purposes.
"""

from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field


FileType = Literal["csv", "excel", "json", "pdf", "image", "text", "xml", "unknown"]


@dataclass
class ColumnMetadata:
    """Metadata for a single column/field in the analyzed file."""
    
    # Required fields
    name: str
    data_type: str

    # Optional fields
    english_name: Optional[str] = None
    translation_used: bool = False
    null_count: int = 0
    unique_count: int = 0
    max_length: Optional[int] = None
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class FileStructureInfo:
    """Basic structural information about the file."""
    
    file_name: str
    file_type: FileType
    file_path: str
    file_size_bytes: Optional[int] = None
    total_rows: Optional[int] = None
    total_columns: Optional[int] = None
    sheet_names: Optional[List[str]] = None  # For Excel files


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics."""
    
    total_null_values: int = 0
    null_percentage: float = 0.0
    duplicate_rows: int = 0
    potential_issues: List[str] = field(default_factory=list)


@dataclass
class FileAnalysisResult:
    """
    Complete result from analyzing a file for database ingestion.
    
    This is the primary model that should be stored in WorkflowState
    instead of a raw string.
    """
    
    # Core structure information
    structure: FileStructureInfo
    
    # Column/field details
    columns: List[ColumnMetadata] = field(default_factory=list)
    
    # Data quality assessment
    quality_metrics: DataQualityMetrics = field(default_factory=DataQualityMetrics)
    
    # Raw content samples
    content_preview: Optional[str] = None
    sample_data: Optional[List[Dict[str, Any]]] = None
    
    # Analysis metadata
    analysis_timestamp: Optional[str] = None
    analysis_success: bool = True
    error_message: Optional[str] = None
    
    # Additional context
    detected_patterns: List[str] = field(default_factory=list)
    suggested_transformations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "structure": {
                "file_name": self.structure.file_name,
                "file_type": self.structure.file_type,
                "file_path": self.structure.file_path,
                "file_size_bytes": self.structure.file_size_bytes,
                "total_rows": self.structure.total_rows,
                "total_columns": self.structure.total_columns,
                "sheet_names": self.structure.sheet_names
            },
            "columns": [
                {
                    "name": col.name,
                    "data_type": col.data_type,
                    "null_count": col.null_count,
                    "unique_count": col.unique_count,
                    "max_length": col.max_length,
                    "sample_values": col.sample_values[:3]  # Limit samples
                }
                for col in self.columns
            ],
            "quality_metrics": {
                "total_null_values": self.quality_metrics.total_null_values,
                "null_percentage": self.quality_metrics.null_percentage,
                "duplicate_rows": self.quality_metrics.duplicate_rows,
                "potential_issues": self.quality_metrics.potential_issues
            },
            "content_preview": self.content_preview,
            "sample_data": self.sample_data[:3] if self.sample_data else None,
            "analysis_timestamp": self.analysis_timestamp,
            "analysis_success": self.analysis_success,
            "error_message": self.error_message,
            "detected_patterns": self.detected_patterns,
            "suggested_transformations": self.suggested_transformations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileAnalysisResult':
        """Create instance from dictionary."""
        structure_data = data.get("structure", {})
        structure = FileStructureInfo(
            file_name=structure_data.get("file_name", ""),
            file_type=structure_data.get("file_type", "unknown file_type"),
            file_path=structure_data.get("file_path", ""),
            file_size_bytes=structure_data.get("file_size_bytes"),
            total_rows=structure_data.get("total_rows"),
            total_columns=structure_data.get("total_columns"),
            sheet_names=structure_data.get("sheet_names")
        )
        
        columns = [
            ColumnMetadata(
                name=col.get("name", ""),
                data_type=col.get("data_type", "unknown data type"),
                null_count=col.get("null_count", 0),
                unique_count=col.get("unique_count", 0),
                max_length=col.get("max_length"),
                sample_values=col.get("sample_values", [])
            )
            for col in data.get("columns", [])
        ]
        
        quality_data = data.get("quality_metrics", {})
        quality_metrics = DataQualityMetrics(
            total_null_values=quality_data.get("total_null_values", 0),
            null_percentage=quality_data.get("null_percentage", 0.0),
            duplicate_rows=quality_data.get("duplicate_rows", 0),
            potential_issues=quality_data.get("potential_issues", [])
        )
        
        return cls(
            structure=structure,
            columns=columns,
            quality_metrics=quality_metrics,
            content_preview=data.get("content_preview"),
            sample_data=data.get("sample_data"),
            analysis_timestamp=data.get("analysis_timestamp"),
            analysis_success=data.get("analysis_success", True),
            error_message=data.get("error_message"),
            detected_patterns=data.get("detected_patterns", []),
            suggested_transformations=data.get("suggested_transformations", [])
        )
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the analysis."""
        lines = [
            f"File Analysis Summary: {self.structure.file_name}",
            f"Type: {self.structure.file_type}",
        ]
        
        if self.structure.total_rows:
            lines.append(f"Rows: {self.structure.total_rows:,}")
        if self.structure.total_columns:
            lines.append(f"Columns: {self.structure.total_columns}")
        
        if self.columns:
            lines.append(f"\nColumn Details:")
            for col in self.columns[:5]:
                lines.append(f"  • {col.name}: {col.data_type}")
        
        if self.quality_metrics.potential_issues:
            lines.append(f"\nQuality Issues: {len(self.quality_metrics.potential_issues)}")
        
        return "\n".join(lines)


@dataclass  
class CSVAnalysisResult(FileAnalysisResult):
    """Specialized result for CSV file analysis."""
    
    delimiter: str = ","
    encoding: str = "utf-8"
    has_header: bool = True


@dataclass
class ExcelAnalysisResult(FileAnalysisResult):
    """Specialized result for Excel file analysis."""
    
    active_sheet: Optional[str] = None
    sheet_analyses: Dict[str, 'FileAnalysisResult'] = field(default_factory=dict)


@dataclass
class ImageAnalysisResult(FileAnalysisResult):
    """Specialized result for image/document analysis."""
    
    extracted_text: Optional[str] = None
    document_type: Optional[str] = None  # "invoice", "form", "receipt", etc.
    ocr_confidence: Optional[float] = None
    structured_fields: Dict[str, Any] = field(default_factory=dict)