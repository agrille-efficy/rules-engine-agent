"""
Domain classification for data files using pattern matching.
"""
import logging


class DomainClassifier:
    """Classifies data domains based on column names with multilingual support."""
    
    def __init__(self):
        # Multilingual domain detection patterns
        self.domain_patterns = {
            'leads': [
                'lead', 'prospect', 'lead_status', 'source', 'qualification', 'score', 'conversion',
                'prospectus', 'candidat', 'piste'
            ],
            'opportunities': [
                'opportunity', 'deal', 'pipeline', 'stage', 'probability', 'close_date', 'forecast',
                'opportunité', 'affaire', 'vente', 'étape', 'probabilité'
            ],
            'contacts': [
                'contact', 'person', 'individual', 'first_name', 'last_name', 'title',
                'personne', 'individu', 'prénom', 'nom', 'téléphone', 'courriel'
            ],
            'companies': [
                'company', 'organization', 'enterprise', 'business', 'industry', 'sector',
                'société', 'entreprise', 'organisation', 'industrie', 'secteur'
            ],
            'activities': [
                'activity', 'action', 'event', 'log', 'history', 'timeline', 'interaction',
                'activité', 'événement', 'historique', 'interaction', 'visite'
            ],
            'communication': [
                'mail', 'email', 'message', 'subject', 'body', 'sender', 'recipient',
                'expéditeur', 'destinataire', 'objet', 'corps', 'visite'
            ]
        }
        
        # Domain metadata mapping
        self.domain_mapping = {
            'leads': {
                'primary_domain': 'lead management and prospecting', 
                'business_area': 'sales lead generation', 
                'data_category': 'leads',
                'table_hints': ['lead', 'prospect', 'qualification']
            },
            'opportunities': {
                'primary_domain': 'sales opportunity tracking', 
                'business_area': 'sales pipeline management', 
                'data_category': 'opportunities',
                'table_hints': ['opportunity', 'deal', 'sales_pipeline']
            },
            'contacts': {
                'primary_domain': 'contact and person management', 
                'business_area': 'relationship management', 
                'data_category': 'contacts',
                'table_hints': ['contact', 'person', 'individual']
            },
            'companies': {
                'primary_domain': 'company and organization management', 
                'business_area': 'corporate data management', 
                'data_category': 'companies',
                'table_hints': ['company', 'organization', 'enterprise']
            },
            'activities': {
                'primary_domain': 'activity and event tracking', 
                'business_area': 'interaction management', 
                'data_category': 'activities',
                'table_hints': ['activity', 'event', 'action', 'visit']
            },
            'communication': {
                'primary_domain': 'communication and messaging', 
                'business_area': 'correspondence', 
                'data_category': 'communication',
                'table_hints': ['mail', 'email', 'message', 'visit']
            },
            'general': {
                'primary_domain': 'business data', 
                'business_area': 'general operations', 
                'data_category': 'business', 
                'table_hints': ['data', 'general']
            }
        }
    
    def infer_domain(self, columns):
        """
        Infer data domain from column names with multilingual support.
        
        Args:
            columns: List of column names
            
        Returns:
            dict: Domain information with metadata
        """
        columns_lower = [col.lower() for col in columns]
        
        # Calculate domain scores with weighted importance
        domain_scores = {}
        for domain, keywords in self.domain_patterns.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                for col in columns_lower:
                    if keyword == col:
                        score += 5
                        matched_keywords.append(keyword)
                    elif keyword in col:
                        if domain == 'communication' and keyword in ['expéditeur', 'destinataire', 'objet']:
                            score += 4
                        elif keyword.endswith('_id'):
                            score += 3
                        else:
                            score += 2
                        matched_keywords.append(keyword)
                        break
            
            domain_scores[domain] = {
                'score': score,
                'matched_keywords': list(set(matched_keywords))
            }
        
        # Get best matching domain
        best_domain = max(domain_scores, key=lambda x: domain_scores[x]['score']) if domain_scores else 'general'
        best_score = domain_scores[best_domain]['score'] if best_domain != 'general' else 0
        
        result = self.domain_mapping.get(best_domain, self.domain_mapping['general'])
        
        # Add detection metadata
        result['detection_confidence'] = min(best_score / 5.0, 1.0)
        result['matched_keywords'] = domain_scores.get(best_domain, {}).get('matched_keywords', [])
        result['all_scores'] = {k: v['score'] for k, v in domain_scores.items() if v['score'] > 0}
        
        return result
    
    def detect_relationship_data(self, columns):
        """
        Detect if data appears to be relationship/mapping data.
        
        Args:
            columns: List of column names
            
        Returns:
            bool: True if relationship data detected
        """
        columns_lower = [col.lower() for col in columns]
        
        # Indicators of relationship data
        relationship_indicators = [
            'mapping', 'relation', 'link', 'association', 'junction',
            '_id', 'id_', 'ref_', '_ref', 'key_', '_key', 'K'
        ]
        
        relationship_score = 0
        total_columns = len(columns)
        
        for col in columns_lower:
            for indicator in relationship_indicators:
                if indicator in col:
                    relationship_score += 1
                    break
        
        id_columns = sum(1 for col in columns_lower if col.endswith('_id') or col.startswith('id_') or 'ref' in col)
        id_ratio = id_columns / max(total_columns, 1)
        
        descriptive_fields = ['name', 'title', 'description', 'email', 'phone', 'address', 'date']
        descriptive_count = sum(1 for col in columns_lower for field in descriptive_fields if field in col)
        
        is_relationship_data = (
            (relationship_score >= 2) or
            (id_ratio > 0.5 and total_columns <= 5) or
            (id_ratio > 0.7) or
            (descriptive_count == 0 and total_columns <= 4)
        )
        
        logging.info(f"Relationship detection - Score: {relationship_score}, ID ratio: {id_ratio:.2f}, "
                    f"Descriptive fields: {descriptive_count}, Decision: {is_relationship_data}")
        
        return is_relationship_data
