"""
Entity-First RAG Pipeline for database table matching.
Orchestrates file analysis, semantic search, and table selection.
"""
import os
import json
import pandas as pd
import logging
from typing import Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from ..config import get_settings
from ..models.validators import (
    PipelineInput, 
    UserContextInput, 
    validate_output_path,
    sanitize_for_logging
)
from .memory import EntityRelationMemory
from ..services.translator import UniversalTranslator
from .domain_classifier import DomainClassifier
from .query_generator import QueryGenerator
from .vector_search import VectorSearchService
from .utils import calculate_confidence_level

settings = get_settings()

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=settings.temperature
    )

def _get_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key
    )

def _get_vector_store():
    from qdrant_client import QdrantClient
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Dependencies loaded successfully.")
logging.info("Environment variables loaded successfully.")

# Initialize clients
openai_client = _get_llm()
embeddings = _get_embeddings_model()
qdrant_client = _get_vector_store()

logging.info("OpenAI and Qdrant clients initialized successfully.")


class GenericFileIngestionRAGPipeline:
    """
    Generic RAG pipeline with Entity-First architecture for analyzing any data file 
    and finding the best database tables using two-stage semantic search.
    """
    
    def __init__(self, qdrant_client, embeddings, collection_name, query_only=True):
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.llm_client = _get_llm()
        self.query_only = query_only
        
        # Initialize components
        self.memory = EntityRelationMemory()
        self.translator = UniversalTranslator()
        self.domain_classifier = DomainClassifier()
        self.query_generator = QueryGenerator(self.translator)
        self.vector_search = VectorSearchService(qdrant_client, embeddings, collection_name)
        
        # Verify collection exists in query-only mode
        if self.query_only:
            existing_collections = [col.name for col in self.qdrant_client.get_collections().collections]
            if self.collection_name not in existing_collections:
                raise ValueError(f"Collection '{self.collection_name}' does not exist. Please run the RAG system in 'feed' mode first to populate the vector store.")
            logging.info(f"Query-only mode: Using existing collection '{self.collection_name}'")
    
    def run_entity_first_pipeline(self, file_analysis, user_context: Optional[str] = None):
        """
        Run the complete Entity-First two-stage pipeline.
        
        Args:
            file_analysis: FileAnalysisResult object with structured file data
            user_context: Optional user context string (will be sanitized)
            
        Returns:
            Dict with entity/relation search results
            
        Raises:
            ValueError: If inputs contain malicious patterns
        """
        # Sanitize user context to prevent prompt injection
        safe_user_context = None
        if user_context and user_context.strip():
            try:
                validated_context = UserContextInput(raw_context=user_context)
                safe_user_context = validated_context.get_sanitized()
                logging.info(f"User context validated and sanitized: {sanitize_for_logging(safe_user_context, 50)}")
            except ValueError as e:
                logging.error(f"User context validation failed: {e}")
                raise ValueError(f"Invalid user context: {e}")
        
        logging.info("=== ENTITY-FIRST RAG PIPELINE ===")
        logging.info(f"Analyzing: {sanitize_for_logging(file_analysis.structure.file_name)}")
        
        logging.info(f"{file_analysis.structure.file_type} file: {file_analysis.structure.total_columns} columns, {file_analysis.structure.total_rows} rows")
        
        # Detect if this is relationship data
        columns = [col.name for col in file_analysis.columns]
        is_relationship_data = self.domain_classifier.detect_relationship_data(columns)
        if is_relationship_data:
            self.memory.set_relationship_data_flag(True)
            logging.info("Detected potential relationship/mapping data")
        
        # Generate semantic search queries (query_generator now handles validation internally)
        logging.info("Step 3: Generating semantic search queries...")
        queries = self.query_generator.generate_queries(file_analysis, safe_user_context)
        logging.info(f"Generated {len(queries)} validated queries for database search")
        
        # STAGE 1: Entity-First Search
        logging.info("\n=== STAGE 1: ENTITY-FIRST SEARCH ===")
        entity_results = self.vector_search.search_entity_tables_only(queries)
        logging.info(f"Found {len(entity_results)} entity tables")
        
        ranked_entities = self.vector_search.rank_tables_by_relevance(entity_results)
        
        entities_stored = 0
        for entity in ranked_entities:
            if entity['composite_score'] >= self.memory.confidence_threshold_entity:
                self.memory.store_entity(entity['table_name'], entity, entity['composite_score'])
                entities_stored += 1
        
        logging.info(f"Stored {entities_stored} high-confidence entities (threshold: {self.memory.confidence_threshold_entity})")
        
        # STAGE 2: Relationship Discovery
        logging.info("\n=== STAGE 2: RELATIONSHIP DISCOVERY ===")
        
        if self.memory.relationship_data_flag:
            relation_results = self.vector_search.search_relation_tables_only(queries)
            logging.info(f"Found {len(relation_results)} relation tables for relationship data")
        else:
            best_entity = self.memory.get_best_entity()
            if best_entity:
                relation_results = self.vector_search.search_related_tables(queries, best_entity['table_name'])
                logging.info(f"Found {len(relation_results)} tables related to {best_entity['table_name']}")
            else:
                relation_results = {}
                logging.warning("No high-confidence entities found, skipping relationship discovery")
        
        ranked_relations = []
        if relation_results:
            ranked_relations = self.vector_search.rank_tables_by_relevance(relation_results)
            
            relations_stored = 0
            for relation in ranked_relations:
                if relation['composite_score'] >= self.memory.confidence_threshold_relation:
                    related_entities = self._find_related_entities(relation['table_name'])
                    self.memory.store_relation(relation['table_name'], relation, 
                                             relation['composite_score'], related_entities)
                    relations_stored += 1
            
            logging.info(f"Stored {relations_stored} relations (threshold: {self.memory.confidence_threshold_relation})")
        
        # Ensure critical relationship tables are included even if not in top 15
        # This helps with common patterns like Company, User, Contact relationships
        critical_patterns = ['_comp', '_user', '_cont']
        critical_relations = []
        
        if ranked_relations and ranked_entities:
            # Get the primary entity name (e.g., "Opportunity")
            primary_entity = ranked_entities[0]['table_name'] if ranked_entities else None
            
            if primary_entity:
                entity_prefix = primary_entity[:4].lower()  # e.g., "oppo"
                
                # Find critical relationship tables that match patterns
                for relation in ranked_relations:
                    table_name_lower = relation['table_name'].lower()
                    
                    # Check if this is a critical relationship table for the primary entity
                    if (entity_prefix in table_name_lower and 
                        any(pattern in table_name_lower for pattern in critical_patterns)):
                        critical_relations.append(relation)
                        logging.info(f"Identified critical relationship table: {relation['table_name']} (score: {relation['composite_score']:.3f})")
        
        # Compile final results
        memory_summary = self.memory.get_summary()
        columns = [col.english_name or col.name for col in file_analysis.columns]
        
        # Build stage2_relation_results with critical tables guaranteed
        stage2_results = ranked_relations[:15] if ranked_relations else []
        
        # Add critical tables if they're not already in top 15
        if critical_relations:
            stage2_table_names = {t['table_name'] for t in stage2_results}
            for critical in critical_relations:
                if critical['table_name'] not in stage2_table_names:
                    stage2_results.append(critical)
                    logging.info(f"Added critical table to results: {critical['table_name']}")
        
        final_results = {
            'pipeline_type': 'entity_first_two_stage',
            'file_analysis': {
                'file_name': file_analysis.structure.file_name,
                'file_type': file_analysis.structure.file_type,
                'total_rows': file_analysis.structure.total_rows,
                'total_columns': file_analysis.structure.total_columns,
                'columns': columns,
                'original_columns': [col.name for col in file_analysis.columns],
            },
            'inferred_domain': self.domain_classifier.infer_domain(columns),
            'user_context': safe_user_context,  # Use sanitized context
            'search_queries_used': queries,
            'relationship_data_detected': memory_summary['relationship_data_detected'],
            'entities_discovered': memory_summary['entities_discovered'],
            'relations_discovered': memory_summary['relations_discovered'],
            'memory_summary': memory_summary,
            'ingestion_summary': self._create_entity_first_summary(memory_summary),
            'stage1_entity_results': ranked_entities[:10] if ranked_entities else [],
            'stage2_relation_results': stage2_results,  # Now includes critical tables
            'top_25_tables': (ranked_entities + ranked_relations)[:25]
        }
        
        return final_results
    
    def _find_related_entities(self, relation_table_name):
        """Find which entities are related to a given relation table"""
        related_entities = []
        
        for entity_name in self.memory.entities.keys():
            if entity_name.lower() in relation_table_name.lower():
                related_entities.append(entity_name)
        
        return related_entities
    
    def _create_entity_first_summary(self, memory_summary):
        """Create ingestion summary for entity-first pipeline"""
        best_entity = memory_summary.get('best_entity')
        entities_count = memory_summary.get('entities_discovered', 0)
        relations_count = memory_summary.get('relations_discovered', 0)
        is_relationship_data = memory_summary.get('relationship_data_detected', False)
        
        if is_relationship_data and relations_count > 0:
            best_relations = memory_summary.get('all_relations', [])
            recommended_table = best_relations[0]['table_name'] if best_relations else None
            confidence_level = calculate_confidence_level(best_relations[0]['confidence_score']) if best_relations else 'None'
            table_type = 'relation'
        elif best_entity:
            recommended_table = best_entity['table_name']
            confidence_level = calculate_confidence_level(best_entity['confidence_score'])
            table_type = 'entity'
        else:
            recommended_table = None
            confidence_level = 'None'
            table_type = 'unknown'
        
        requires_review = (
            confidence_level in ['None', 'Low'] or 
            (entities_count == 0 and relations_count == 0)
        )
        
        mapping_ready = (
            confidence_level in ['High', 'Medium'] and 
            recommended_table is not None
        )
        
        return {
            'recommended_table': recommended_table,
            'recommended_table_type': table_type,
            'confidence_level': confidence_level,
            'requires_review': requires_review,
            'mapping_ready': mapping_ready,
            'entities_found': entities_count,
            'relations_found': relations_count,
            'relationship_data_detected': is_relationship_data,
            'pipeline_success': mapping_ready or (entities_count > 0 or relations_count > 0)
        }
    
    def display_results_summary(self, results):
        """Display a formatted summary optimized for agent consumption"""
        if 'error' in results:
            logging.error(f"Pipeline Error: {results['error']}")
            return
        
        file_info = results['file_analysis']
        summary = results['ingestion_summary']
        
        logging.info("=" * 80)
        logging.info(f"INGESTION ANALYSIS: {sanitize_for_logging(file_info['file_name'])}")
        logging.info("=" * 80)
        logging.info(f"File: {file_info['file_type']} | {file_info['total_rows']} rows | {file_info['total_columns']} columns")
        logging.info(f"Domain: {results['inferred_domain']['primary_domain']}")
        logging.info(f"Best Table: {summary['recommended_table']}")
        logging.info(f"Confidence: {summary['confidence_level']}")
        logging.info(f"Review Required: {'Yes' if summary['requires_review'] else 'No'}")
        logging.info(f"Mapping Ready: {'Yes' if summary['mapping_ready'] else 'No'}")
        
        logging.info("TOP 25 DATABASE TABLES:")  # Updated from TOP 10
        for i, table in enumerate(results.get('top_25_tables', [])[:25], 1):  # Updated to top_25_tables
            logging.info(f"{i}. {table['table_name']} ({table['table_kind']})")
            logging.info(f"   Score: {table['composite_score']:.3f} | Fields: {table['field_count']} | Matches: {table['total_matches']}")
                
        if summary['mapping_ready']:
            logging.info("Ready for Mapping Processing")
        else:
            logging.warning("Requires review before mapping generation")
    
    def export_for_sql_agent(self, results, output_file=None):
        """Export results in format optimized for SQL generation agent"""
        if 'error' in results:
            return results
        
        if not output_file:
            file_name = results['file_analysis']['file_name']
            base_name = os.path.splitext(file_name)[0]
            # Sanitize base_name to prevent path injection
            base_name = base_name.replace('..', '').replace('/', '_').replace('\\', '_')
            output_file = rf"results_for_agent\{base_name}_ingestion_analysis.json"
        
        # Validate output path to prevent path traversal attacks
        try:
            safe_output_path = validate_output_path(output_file)
            logging.info(f"Validated output path: {safe_output_path}")
        except ValueError as e:
            error_msg = f"Invalid output path: {e}"
            logging.error(error_msg)
            return {'error': error_msg}
        
        sql_agent_data = {
            'source_file': results['file_analysis']['file_name'],
            'file_structure': {
                'columns': results['file_analysis']['columns'],
                'total_rows': results['file_analysis']['total_rows']
            },
            'recommended_ingestion': {
                'primary_table': results['ingestion_summary']['recommended_table'],
                'confidence': results['ingestion_summary']['confidence_level'],
                'ready_for_sql': results['ingestion_summary']['mapping_ready']
            },
            'table_options': [
                {
                    'table_name': table['table_name'],
                    'table_code': table['table_code'],
                    'composite_score': round(table['composite_score'], 3),
                    'field_count': table['field_count'],
                    'table_schema': table['content']
                }
                for table in results.get('top_25_tables', [])[:25]
            ],
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            with open(safe_output_path, 'w', encoding='utf-8') as f:
                json.dump(sql_agent_data, f, indent=2, ensure_ascii=False)
            logging.info(f"SQL Agent data exported to: {safe_output_path}")
            return {'success': True, 'output_file': str(safe_output_path)}
        except Exception as e:
            return {'error': f'Export failed: {str(e)}'}


logging.info("GenericFileIngestionRAGPipeline class complete with all methods")


def main():
    """Main function with mode selection"""
    import sys
    from .dico_client import DicoAPI
    from .vector_store_builder import feed_vector_store
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <mode>")
        print("Modes:")
        print("  feed   - Fetch DICO data and populate vector store")
        print("Example:")
        print("  python pipeline.py feed")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode != 'feed':
        print("Error: Only 'feed' mode is supported")
        sys.exit(1)
    
    dico_getter = DicoAPI(
        base_url="https://sandbox-5.efficytest.cloud",
        customer="SANDBOX05"
    )
    
    collection_name = "maxo_vector_store_v2"
    
    success = feed_vector_store(dico_getter, qdrant_client, embeddings, collection_name)
    if success:
        print("Vector store feeding completed successfully!")
    else:
        print("Vector store feeding failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()