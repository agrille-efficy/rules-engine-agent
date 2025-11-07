"""
Vector search operations for RAG pipeline.
"""
import logging
from qdrant_client import models


class VectorSearchService:
    """Handles all vector search operations against Qdrant."""
    
    def __init__(self, qdrant_client, embeddings, collection_name):
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name
    
    def search_all_tables(self, queries, top_k=10):
        """
        Search for relevant tables using multiple semantic queries.
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            
        Returns:
            dict: Mapping of table_name -> result data
        """
        all_results = {}
        
        for i, query in enumerate(queries):
            logging.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                query_embedding = self.embeddings.embed_query(query)
                
                search_results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chunk_type",
                                match=models.MatchValue(value="table_ingestion_profile")
                            )
                        ]
                    ),
                    limit=top_k
                )
                
                for point in search_results.points:
                    table_name = point.payload['primary_table']
                    
                    if (table_name not in all_results) or (point.score > all_results[table_name]['score']):
                        all_results[table_name] = {
                            'table_name': table_name,
                            'table_code': point.payload['table_code'],
                            'table_kind': point.payload['table_kind'],
                            'field_count': point.payload['field_count'],
                            'content': point.payload['content'],
                            'metadata': point.payload['metadata'],
                            'score': point.score,
                            'queries_matched': [i+1]
                        }
                    else:
                        all_results[table_name]['queries_matched'].append(i+1)
                    
            except Exception as e:
                logging.error(f"Search error for query {i+1}: {e}")
        
        return all_results
    
    def search_entity_tables_only(self, queries, top_k=10):
        """
        Stage 1: Search only Entity tables.
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            
        Returns:
            dict: Mapping of table_name -> result data
        """
        all_results = {}
        
        for i, query in enumerate(queries):
            try:
                query_embedding = self.embeddings.embed_query(query)
                
                search_results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chunk_type",
                                match=models.MatchValue(value="table_ingestion_profile")
                            ),
                            models.FieldCondition(
                                key="table_kind",
                                match=models.MatchValue(value="Entity")
                            )
                        ]
                    ),
                    limit=top_k
                )
                
                for point in search_results.points:
                    table_name = point.payload['primary_table']
                    
                    if (table_name not in all_results) or (point.score > all_results[table_name]['score']):
                        all_results[table_name] = {
                            'table_name': table_name,
                            'table_code': point.payload['table_code'],
                            'table_kind': point.payload['table_kind'],
                            'field_count': point.payload['field_count'],
                            'content': point.payload['content'],
                            'metadata': point.payload['metadata'],
                            # 'fields': point.payload['metadata'].get('fields', []),
                            'score': point.score,
                            'queries_matched': [i+1]
                        }
                    else:
                        all_results[table_name]['queries_matched'].append(i+1)
                        
            except Exception as e:
                logging.error(f"Entity search error for query {i+1}: {e}")
        
        return all_results
    
    def search_relation_tables_only(self, queries, top_k=10):
        """
        Stage 2a: Search only Relation tables.
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            
        Returns:
            dict: Mapping of table_name -> result data
        """
        all_results = {}
        
        for i, query in enumerate(queries):
            try:
                query_embedding = self.embeddings.embed_query(query)
                
                search_results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chunk_type",
                                match=models.MatchValue(value="table_ingestion_profile")
                            ),
                            models.FieldCondition(
                                key="table_kind",
                                match=models.MatchValue(value="Relation")
                            )
                        ]
                    ),
                    limit=top_k
                )
                
                for point in search_results.points:
                    table_name = point.payload['primary_table']
                    
                    if (table_name not in all_results) or (point.score > all_results[table_name]['score']):
                        all_results[table_name] = {
                            'table_name': table_name,
                            'table_code': point.payload['table_code'],
                            'table_kind': point.payload['table_kind'],
                            'field_count': point.payload['field_count'],
                            'content': point.payload['content'],
                            'metadata': point.payload['metadata'],
                            'score': point.score,
                            'queries_matched': [i+1]
                        }
                    else:
                        all_results[table_name]['queries_matched'].append(i+1)
                        
            except Exception as e:
                logging.error(f"Relation search error for query {i+1}: {e}")
        
        return all_results
    
    def search_related_tables(self, queries, entity_table_name, top_k=10):
        """
        Stage 2b: Search for related tables based on selected entity.
        
        Args:
            queries: List of query strings
            entity_table_name: Name of the primary entity table
            top_k: Number of top results per query
            
        Returns:
            dict: Mapping of table_name -> result data
        """
        all_results = {}
        
        entity_enhanced_queries = queries + [
            f"tables related to {entity_table_name} entity for data relationships",
            f"junction tables connecting {entity_table_name} to other entities"
        ]
        
        for i, query in enumerate(entity_enhanced_queries):
            try:
                query_embedding = self.embeddings.embed_query(query)
                
                search_results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chunk_type",
                                match=models.MatchValue(value="table_ingestion_profile")
                            )
                        ]
                    ),
                    limit=top_k
                )
                
                for point in search_results.points:
                    table_name = point.payload['primary_table']
                    
                    if table_name == entity_table_name:
                        continue
                    
                    if (table_name not in all_results) or (point.score > all_results[table_name]['score']):
                        all_results[table_name] = {
                            'table_name': table_name,
                            'table_code': point.payload['table_code'],
                            'table_kind': point.payload['table_kind'],
                            'field_count': point.payload['field_count'],
                            'content': point.payload['content'],
                            'metadata': point.payload['metadata'],
                            'score': point.score,
                            'queries_matched': [i+1]
                        }
                    else:
                        all_results[table_name]['queries_matched'].append(i+1)
                        
            except Exception as e:
                logging.error(f"Related search error for query {i+1}: {e}")
        
        return all_results
    
    def rank_tables_by_relevance(self, search_results):
        """
        Rank tables by multiple relevance criteria.
        
        Args:
            search_results: Dict of table_name -> result data
            
        Returns:
            list: Ranked list of table results
        """
        ranked_tables = []
        
        for table_name, data in search_results.items():
            # Extract fields from known locations to preserve them through ranking
            fields = data.get('fields') or data.get('metadata', {}).get('fields')
            if not fields and 'best_chunks' in data:
                # Try to extract from multi-level best_chunks metadata
                try:
                    fields = data['best_chunks'].get('table_ingestion_profile', {}).get('metadata', {}).get('fields')
                except Exception:
                    fields = None
            if not fields:
                fields = []

            # Handle both search methods' return formats
            if 'scores' in data:
                # Multi-level search format
                scores = data['scores']
                avg_score = sum(scores) / len(scores) if scores else 0
                max_score = max(scores) if scores else 0
                query_coverage = len(set(data['queries_matched']))
                normalized_query_coverage = min(query_coverage / 10.0, 1.0)
                composite_score = (max_score * 0.5) + (avg_score * 0.3) + (normalized_query_coverage * 0.2)
                total_matches = len(scores)
                content = data['best_chunks'].get('table_ingestion_profile', {}).get('content', '') if 'best_chunks' in data else ''
            else:
                # Simple search format
                score = data['score']
                avg_score = score
                max_score = score
                query_coverage = len(set(data['queries_matched']))
                normalized_query_coverage = min(query_coverage / 10.0, 1.0)
                composite_score = (max_score * 0.5) + (avg_score * 0.3) + (normalized_query_coverage * 0.2)
                total_matches = 1
                content = data['content']
            
            ranked_tables.append({
                'table_name': table_name,
                'table_code': data['table_code'],
                'table_kind': data['table_kind'],
                'field_count': data['field_count'],
                'content': content,
                'avg_score': avg_score,
                'max_score': max_score,
                'query_coverage': query_coverage,
                'composite_score': composite_score,
                'total_matches': total_matches,
                'queries_matched': data['queries_matched'],
                'fields': fields
            })
        
        ranked_tables.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return ranked_tables
