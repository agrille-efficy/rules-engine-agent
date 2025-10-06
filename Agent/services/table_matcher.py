from ..config.settings import get_settings
from ..rag.pipeline import GenericFileIngestionRAGPipeline
from ..models.file_analysis_model import FileAnalysisResult 
from ..models.rag_match_model import TableMatch, TableMatchResult

from typing import Optional
import re


class TableMatcherService:
    def __init__(self):
        settings = get_settings()

        from ..rag.pipeline import _get_embeddings_model, _get_vector_store

        self.qdrant_client = _get_vector_store()
        self.embeddings = _get_embeddings_model()
        self.collection_name = settings.qdrant_collection_name

        self.pipeline = GenericFileIngestionRAGPipeline(
            self.qdrant_client,
            self.embeddings,
            self.collection_name,
            query_only=True
        )

    def find_matching_tables(
            self,
            file_analysis: FileAnalysisResult,
            user_context: Optional[str] = None,
    ) -> TableMatchResult:
        """
        Find database tables that match the analyzed file.

        Args:
            file_analysis: FileAnalysisResult from FileAnalysisService.
            user_context: Optional user context to refine the search.
        """
        rag_results = self.pipeline.run_entity_first_pipeline(
            file_analysis,
            user_context
        )

        return self._convert_to_table_match_result(rag_results, file_analysis)
    
    def _convert_to_table_match_result(
            self, 
            rag_results,
            file_analysis: FileAnalysisResult
    ) -> TableMatchResult:
        """
        Convert RAG pipeline results to TableMatchResult model.

        Args:
            rag_results: Dict from pipeline.run_entity_first_pipeline()
            file_analysis: Orignial file analysis

        Returns:
            Structured TableMatchResult
        """
        matched_tables = []

        # Extract entity results
        entity_results = rag_results.get("stage1_entity_results", [])
        for entity in entity_results:
            matched_tables.append(TableMatch(
                table_name=entity.get("table_name"),
                schema_name=None,
                similarity_score=entity['composite_score'],
                confidence=self._score_to_confidence(entity['composite_score']),
                matching_columns=self._extract_matching_columns(entity),
                reason=f"Entity table match (score: {entity['composite_score']:.2f})",
                metadata={
                    'table_code': entity['table_code'],
                    'table_kind': entity['table_kind'],
                    'field_count': entity['field_count'],
                    'query_coverage': entity['query_coverage'],
                }
            ))

        # Extract relation result
        relation_results = rag_results.get("stage2_relation_results", [])
        for relation in relation_results:
            matched_tables.append(TableMatch(
                table_name=relation['table_name'],
                schema_name=None,
                similarity_score=relation['composite_score'],
                confidence=self._score_to_confidence(relation['composite_score']),
                matching_columns=self._extract_matching_columns(relation),
                reason=f"Relation table match (score: {relation['composite_score']:.2f})",
                metadata={
                    'table_code': relation['table_code'],
                    'table_kind': relation['table_kind'],
                    'field_count': relation['field_count'],
                    'query_coverage': relation['query_coverage'],
                }  
            ))

        search_query = f"Analyzed {file_analysis.structure.file_name}: "
        search_query += f"{file_analysis.structure.total_columns} columns, "
        search_query += f"domain: {rag_results.get('inferred_domain', {}).get('primary_domain', 'unknown')}"

        return TableMatchResult(
            matched_tables=matched_tables,
            search_query=search_query,
            total_candidates=len(matched_tables),
        )
    
    def _score_to_confidence(self, score: float) -> str:
        """Convert similarity score to confidence level."""
        if score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"
        
    def _extract_matching_columns(self, rag_table: dict) -> list:
        """Extract column names from RAG table result."""
        metadata = rag_table.get('metadata', {})
        
        if 'primary_fields' in metadata:
            return metadata['primary_fields']
        
        # Fallback: parse from content
        content = rag_table.get('content', '')
        
        fields_match = re.search(r'Primary fields:\s*([^\n]+)', content)
        if fields_match:
            fields_str = fields_match.group(1)
            columns = [col.strip() for col in fields_str.split(',') if col.strip()]
            return columns
        
        return []