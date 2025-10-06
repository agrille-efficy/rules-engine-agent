"""
Semantic query generation for RAG table matching.
"""
from .domain_classifier import DomainClassifier


class QueryGenerator:
    """Generates semantic queries for database table matching."""
    
    def __init__(self, translator=None):
        self.translator = translator
        self.domain_classifier = DomainClassifier()
    
    def generate_queries(self, file_analysis, user_context=None):
        """
        Generate semantic queries from FileAnalysisResult object.
        
        Args:
            file_analysis: FileAnalysisResult object with structured file data
            user_context: Optional user context string
            
        Returns:
            List of semantic query strings
        """
        # Extract English columns for better semantic matching
        columns = [col.english_name or col.name for col in file_analysis.columns]
        
        # Detect domain using English columns
        domain_hints = self.domain_classifier.infer_domain(columns)
        file_name = file_analysis.structure.file_name
        file_type = file_analysis.structure.file_type
        
        # Translate user context if needed
        if user_context and self.translator:
            user_context = self.translator.translate_domain_context(user_context)
        
        # Enhanced query templates using English column names for better matching
        query_templates = [
            f"database table for storing {domain_hints['primary_domain']} data with fields like {', '.join(columns[:6])}",
            f"{file_type} data ingestion into relational database with columns {', '.join(columns[:8])}",
            f"business data management system for {domain_hints['business_area']} information",
            f"data warehouse table for {domain_hints['data_category']} records and analytics",
            f"structured data storage for {file_name} content in enterprise database"
        ]

        # Add domain-specific targeted queries
        query_templates.extend(self._get_domain_specific_queries(domain_hints['data_category']))
        
        if user_context:
            context_query = f"{user_context} with data structure: {', '.join(columns[:10])}"
            query_templates.insert(0, context_query)
        
        return query_templates
    
    def _get_domain_specific_queries(self, data_category):
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
