"""
Data models for the Agent workflow system.
"""

from .workflow_state import WorkflowState
from .file_analysis_model import (
    FileAnalysisResult,
    FileStructureInfo,
    DataQualityMetrics,
    CSVAnalysisResult,
    ExcelAnalysisResult,
    ImageAnalysisResult,
    FileType
)


# Define public API
__all__ = [
    "WorkflowState",

    "FileAnalysisResult", 
    "FileStructureInfo",
    "DataQualityMetrics",
    "CSVAnalysisResult",
    "ExcelAnalysisResult",
    "ImageAnalysisResult",
    "FileType"
]
