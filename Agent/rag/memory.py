class EntityRelationMemory:
    """Memory storage for discovered entities and relations across the two-stage pipeline"""
    
    def __init__(self):
        self.entities = {}
        self.relations = {}
        self.relationship_data_flag = False
        self.confidence_threshold_entity = 0.6  # Higher threshold for entities
        self.confidence_threshold_relation = 0.6  # Standard threshold for relations
        
    def store_entity(self, table_name, table_data, confidence_score):
        """Store discovered entity table and its metadata"""
        self.entities[table_name] = {
            'table_name': table_name,
            'table_code': table_data.get('table_code'),
            'table_kind': table_data.get('table_kind'),
            'field_count': table_data.get('field_count'),
            'content': table_data.get('content'),
            'confidence_score': confidence_score,
            'composite_score': table_data.get('composite_score'),
            'discovered_stage': 'entity_first'
        }
        logging.info(f"Stored entity: {table_name} with confidence {confidence_score}")
    
    def store_relation(self, table_name, table_data, confidence_score, related_entities=None):
        """Store discovered relation table and its dependencies"""
        self.relations[table_name] = {
            'table_name': table_name,
            'table_code': table_data.get('table_code'),
            'table_kind': table_data.get('table_kind'),
            'field_count': table_data.get('field_count'),
            'content': table_data.get('content'),
            'confidence_score': confidence_score,
            'composite_score': table_data.get('composite_score'),
            'related_entities': related_entities or [],
            'discovered_stage': 'relationship_discovery'
        }
        logging.info(f"Stored relation: {table_name} with confidence {confidence_score}")
    
    def get_best_entity(self):
        """Get the highest confidence entity table"""
        if not self.entities:
            return None
        return max(self.entities.values(), key=lambda x: x['confidence_score'])
    
    def get_all_entities(self):
        """Get all discovered entity tables sorted by confidence"""
        return sorted(self.entities.values(), key=lambda x: x['confidence_score'], reverse=True)
    
    def get_all_relations(self):
        """Get all discovered relation tables sorted by confidence"""
        return sorted(self.relations.values(), key=lambda x: x['confidence_score'], reverse=True)
    
    def set_relationship_data_flag(self, flag=True):
        """Flag indicating source data appears to be relationship/mapping data"""
        self.relationship_data_flag = flag
        logging.info(f"Relationship data flag set to: {flag}")
    
    def get_summary(self):
        """Get complete summary of discovered tables"""
        return {
            'entities_discovered': len(self.entities),
            'relations_discovered': len(self.relations),
            'relationship_data_detected': self.relationship_data_flag,
            'best_entity': self.get_best_entity(),
            'all_entities': self.get_all_entities(),
            'all_relations': self.get_all_relations(),
            'entity_confidence_threshold': self.confidence_threshold_entity,
            'relation_confidence_threshold': self.confidence_threshold_relation
        }