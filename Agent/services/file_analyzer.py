"""
File analysis service - bridges tools and models.
Structures CSV file analysis results into data models with translation support.
Currently supports CSV files only.
"""
import json
import os
from datetime import datetime
from typing import List
from ..tools.file_tools import analyze_file
from ..services.translator import UniversalTranslator
from ..models.file_analysis_model import (
    CSVAnalysisResult,
    FileStructureInfo,
    ColumnMetadata,
    DataQualityMetrics
)

class FileAnalyzerService:
    """
    Service that analyzes CSV files and provides structured outputs with translation.
    This is not a LangChain tool - it's internal business logic.
    """
    def __init__(self):
        self.translator = UniversalTranslator()

    def analyze(self, file_path: str) -> CSVAnalysisResult:
        """
        Analyze CSV file and return structured result with translated column names.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            CSVAnalysisResult object with both original and English column names
        """
        # Call the tool - returns JSON string for CSV
        result = analyze_file.invoke({"file_path": file_path})

        try:
            data = json.loads(result)
            
            # Check for errors in response
            if data.get("error"):
                return self._create_error_result(
                    file_path, 
                    data.get("error", "Unknown error")
                )
            
            # Parse JSON result with translation
            return self._parse_csv_json(data, file_path)
            
        except json.JSONDecodeError as e:
            return self._create_error_result(file_path, f"Failed to parse tool output: {str(e)}")
        except Exception as e:
            return self._create_error_result(file_path, f"Analysis failed: {str(e)}")

    def _parse_csv_json(self, data: dict, file_path: str) -> CSVAnalysisResult:
        """
        Parse JSON output from analyze_file tool into CSVAnalysisResult.
        Includes column translation for better semantic matching.
        """
        # Extract column names
        original_columns = data.get("columns", [])
        
        # Translate column names to English for better RAG matching
        english_columns, translation_mapping = self.translator.translate_column_names(original_columns)
        
        # Build column metadata with both original and English names
        columns = []
        for col_data in data.get("data_types", []):
            col_name = col_data["name"]
            col_index = original_columns.index(col_name) if col_name in original_columns else -1
            english_name = english_columns[col_index] if col_index >= 0 else col_name
            
            column = ColumnMetadata(
                name=col_name,                           # Original name
                english_name=english_name,               # Translated name
                translation_used=(col_name != english_name),  # Translation flag
                data_type=col_data.get("data_type", "Unknown data type"),
                max_length=col_data.get("max_length"),
                null_count=col_data.get("null_count", 0),
                unique_count=col_data.get("unique_count", 0),
                sample_values=col_data.get("sample_values", [])
            )
            columns.append(column)

        # Create structure
        dimensions = data.get("dimensions", {})
        structure = FileStructureInfo(
            file_name=data.get("file_name", os.path.basename(file_path)),
            file_type="csv",
            file_path=file_path,
            file_size_bytes=os.path.getsize(file_path) if os.path.exists(file_path) else None,
            total_rows=dimensions.get("rows", 0),
            total_columns=dimensions.get("columns", 0)
        )

        # Calculate quality metrics
        total_nulls = sum(col.null_count for col in columns)
        total_cells = structure.total_rows * structure.total_columns

        quality_metrics = DataQualityMetrics(
            total_null_values=total_nulls,
            null_percentage=(total_nulls / total_cells * 100) if total_cells > 0 else 0.0,
            duplicate_rows=0,  # Not provided by tool yet
            potential_issues=self._detect_quality_issues(columns, structure)
        )

        return CSVAnalysisResult(
            structure=structure,
            columns=columns,
            quality_metrics=quality_metrics,
            sample_data=data.get("sample_data", [])[:5],
            delimiter=data.get("delimiter", ","),
            encoding=data.get("encoding", "utf-8"),
            has_header=True,
            content_preview=json.dumps(data, indent=2)[:1000],
            analysis_success=True,
            analysis_timestamp=datetime.now().isoformat()
        )

    def _detect_quality_issues(self, columns: List[ColumnMetadata], structure: FileStructureInfo) -> List[str]:
        """Detect data quality issues in analyzed columns."""
        issues = []

        for col in columns: 
            if structure.total_rows and col.null_count > 0:
                null_percentage = (col.null_count / structure.total_rows) * 100
                if null_percentage > 50:
                    issues.append(f"Column '{col.name}' has {null_percentage:.1f}% null values")

            if col.unique_count == 1 and structure.total_rows > 1:
                issues.append(f"Column '{col.name}' has only one unique value")
                
            if col.unique_count == structure.total_rows and structure.total_rows > 10:
                issues.append(f"Column '{col.name}' has all unique values (potential ID field)")
        
        return issues
    
    def _create_error_result(self, file_path: str, error_message: str) -> CSVAnalysisResult:
        """Create error result when analysis fails."""
        file_name = os.path.basename(file_path)
        
        structure = FileStructureInfo(
            file_name=file_name,
            file_type="csv",
            file_path=file_path
        )
        
        return CSVAnalysisResult(
            structure=structure,
            columns=[],
            quality_metrics=DataQualityMetrics(),
            analysis_timestamp=datetime.now().isoformat(),
            analysis_success=False,
            error_message=error_message
        )
