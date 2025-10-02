"""
File analysis service - bridges tools and models.
Structures file analysis results into data models, making them accessible for other componentns.
"""
import json
from datetime import datetime
from typing import List
from ..tools.file_tools import analyze_file 
from ..models.file_analysis_model import (
    FileAnalysisResult,
    CSVAnalysisResult,
    FileStructureInfo,
    ColumnMetadata,
    DataQualityMetrics
)

class FileAnalyzerService:
    """
    Service that wraps tools and provides structured outpus.
    This is not a LangChain tool - it's internal businell logic.
    """

    def analyze(self, file_path: str) -> FileAnalysisResult:
        """
        Analyze the file and return structured result.
        
        Returns:
            FileAnalysisResult object
        """
        result = analyze_file.invoke({"file_path": file_path})

        try:
            data = json.loads(result)
            return self._parse_json_result(data, file_path)
        except json.JSONDecodeError:
            return self._parse_text_result(result, file_path)

    def _parse_json_result(self, result: json, file_path: str) -> FileAnalysisResult:
        """Parse tool's string output into FileAnalysisResult."""
        structure = FileStructureInfo(
            file_name=result["file_name"],
            file_type=result["file_type"],
            file_path=file_path,
            total_rows=result["dimensions"]["rows"],
            total_columns=result["dimensions"]["columns"]
        )
        
        columns = []
        for col_data in result["data_types"]:
            column = ColumnMetadata(
                name=col_data["name"],
                data_type=col_data["data_type"],
                max_length=col_data.get("max_length"),
                null_count=col_data["null_count"],
                unique_count=col_data["unique_count"],
                sample_values=col_data.get("sample_values", [])
            )
            columns.append(column)

        total_nulls = sum(col.null_count for col in columns)
        total_cells = structure.total_rows * structure.total_columns

        quality_metrics = DataQualityMetrics(
            total_null_values=total_nulls,
            null_percentage=(total_nulls / total_cells * 100) if total_cells > 0 else 0.0,
            potential_issues=self._detect_quality_issues(columns, structure)
        )

        return CSVAnalysisResult(
            structure=structure,
            columns=columns,
            quality_metrics=quality_metrics,
            sample_data=result.get("sample_data", [])[:5],
            delimiter=result.get("delimiter"),
            encoding=result.get("encoding"),
            has_header=True,
            analysis_success=True,
            analysis_timestamp=datetime.now().isoformat()
        )
    

    def _detect_quality_issues(self, columns: List[ColumnMetadata], structure: FileStructureInfo) -> List[str]:
        issues = []

        for col in columns: 
            if structure.total_rows and col.null_count > 0:
                null_percentage = (col.null_count / structure.total_rows) * 100 if structure.total_rows > 0 else 0
                if null_percentage > 50:
                    issues.append(f"Column '{col.name}' has {null_percentage:.2f}% null values.")

            if col.unique_count == 1 and structure.total_rows > 1:
                issues.append(f"Column '{col.name}' has the same value for all rows.")
        return issues

    def _parse_text_result(self, result_str: str, file_path: str) -> FileAnalysisResult:
        """
        Fallback parser for plain text tool outputs (PDF, images, Excel, etc.).
        Extracts basic information and wraps in FileAnalysisResult.
        """
        import os
        
        # Detect file type from result string
        file_type = "unknown"
        if "PDF" in result_str or "PDF Content Analysis" in result_str:
            file_type = "pdf"
        elif "Excel Analysis" in result_str:
            file_type = "excel"
        elif "Image Analysis" in result_str:
            file_type = "image"
        elif "JSON Analysis" in result_str:
            file_type = "json"
        elif "Text Analysis" in result_str:
            file_type = "text"
        
        # Try to extract file name from result
        file_name = os.path.basename(file_path)
        
        # Create basic structure
        structure = FileStructureInfo(
            file_name=file_name,
            file_type=file_type,
            file_path=file_path,
            file_size_bytes=os.path.getsize(file_path) if os.path.exists(file_path) else None
        )
        
        # Try to extract any structural information from text
        # (This is optional - depends on how much you want to parse)
        detected_patterns = []
        if "tabular data" in result_str.lower():
            detected_patterns.append("Contains tabular data")
        if "header" in result_str.lower():
            detected_patterns.append("Headers detected")
        
        # Check for errors
        analysis_success = True
        error_message = None
        if result_str.startswith("Error"):
            analysis_success = False
            error_message = result_str
        
        # Return generic FileAnalysisResult
        return FileAnalysisResult(
            structure=structure,
            content_preview=result_str[:2000],  # Store first 2000 chars
            analysis_timestamp=datetime.now().isoformat(),
            analysis_success=analysis_success,
            error_message=error_message,
            detected_patterns=detected_patterns
        )