"""
Semantic query generation for RAG table matching.
"""
from typing import Optional, List
from ..models.validators import UserContextInput, ColumnNamesInput, QueryInput
from .domain_classifier import DomainClassifier


class QueryGenerator:
    """Generates semantic queries for database table matching."""
    
    def __init__(self, translator=None):
        self.translator = translator
        self.domain_classifier = DomainClassifier()
    
    def generate_queries(self, file_analysis, user_context: Optional[str] = None) -> List[str]:
        """
        Generate semantic queries from FileAnalysisResult object.
        
        Args:
            file_analysis: FileAnalysisResult object with structured file data
            user_context: Optional user context string (will be sanitized)
            
        Returns:
            List of semantic query strings
            
        Raises:
            ValueError: If inputs contain malicious patterns
        """
        # Extract and validate English columns for better semantic matching
        columns = [col.english_name or col.name for col in file_analysis.columns]
        
        # Validate column names
        validated_columns = ColumnNamesInput(columns=columns)
        safe_columns = validated_columns.columns
        
        # Detect domain using validated English columns
        domain_hints = self.domain_classifier.infer_domain(safe_columns)
        file_name = file_analysis.structure.file_name
        file_type = file_analysis.structure.file_type
        
        # Sanitize and validate user context to prevent prompt injection
        safe_user_context = None
        if user_context and user_context.strip():
            try:
                validated_context = UserContextInput(raw_context=user_context)
                safe_user_context = validated_context.get_sanitized()
                
                # Translate user context if needed
                if self.translator:
                    safe_user_context = self.translator.translate_domain_context(safe_user_context)
            except ValueError as e:
                # Log the validation error but continue without user context
                import logging
                logging.warning(f"User context validation failed: {e}. Continuing without user context.")
                safe_user_context = None
        
        # Enhanced query templates using validated column names for better matching
        query_templates = [
            f"database table for storing {domain_hints['primary_domain']} data with fields like {', '.join(safe_columns[:6])}",
            f"{file_type} data ingestion into relational database with columns {', '.join(safe_columns[:8])}",
            f"business data management system for {domain_hints['business_area']} information",
            f"data warehouse table for {domain_hints['data_category']} records and analytics",
            f"structured data storage for {file_name} content in enterprise database"
        ]

        # Add domain-specific targeted queries
        query_templates.extend(self._get_domain_specific_queries(domain_hints['data_category']))
        
        # Add sanitized user context query if available
        if safe_user_context:
            # Safely concatenate with column data (both already validated)
            context_query = f"{safe_user_context} with data structure: {', '.join(safe_columns[:10])}"
            query_templates.insert(0, context_query)
        
        # Validate all generated queries before returning
        validated_queries = []
        for query in query_templates:
            try:
                validated = QueryInput(query=query)
                validated_queries.append(validated.query)
            except ValueError:
                # Skip invalid queries (shouldn't happen with our templates, but safety check)
                continue
        
        return validated_queries
    
    def _get_domain_specific_queries(self, data_category: str) -> List[str]:
        """
        Get domain-specific queries based on data category.
        
        Args:
            data_category: Category of data (e.g., 'communication', 'contacts')
            
        Returns:
            List of domain-specific query strings
        """
        domain_queries = {
            'communication': [
                "email communication tracking and management system with sender recipient subject message",
                "mail message storage organization with expéditeur destinataire objet corps du message",
                "correspondence interaction history email sent received tracking with french fields"
            ],
            'contacts': [
                "contact person individual management with names emails phones addresses",
                "company organization enterprise business with industry sector headquarters"
            ],
            'companies': [
                "contact person individual management with names emails phones addresses",
                "company organization enterprise business with industry sector headquarters"
            ],
            'opportunities': [
                "sales opportunity deal pipeline stage probability close date",
                "revenue forecasting with expected amount win probability sales cycle"
            ]
        }
        
        return domain_queries.get(data_category, [])
