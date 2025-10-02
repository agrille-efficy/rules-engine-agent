import os 
import json 
import pandas as pd 
import hashlib 
import logging 
import requests 
import time
from functools import wraps

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models 
from qdrant_client.models import PointStruct, PayloadSchemaType 
from langchain_openai import OpenAIEmbeddings, OpenAI 

# Fix relative imports to work both as module and when called from agent
try:
    from .chunk_generator import generate_table_ingestion_chunks
    from .config import config
    from ..embedding_cache import EmbeddingCache
except ImportError:
    try:
        from chunk_generator import generate_table_ingestion_chunks
        from config import config
        from embedding_cache import EmbeddingCache
    except ImportError:
        # Create fallback implementations if imports fail
        def generate_table_ingestion_chunks(dico_data):
            return []
        
        class config:
            efficy_base_url = "https://sandbox-5.efficytest.cloud"
            efficy_customer = "SANDBOX05"
            efficy_username = ""
            efficy_password = ""
        
        class EmbeddingCache:
            def __init__(self, embeddings):
                self.embeddings = embeddings
            
            def embed_query(self, query):
                return self.embeddings.embed_query(query)

# Import for translation functionality
import re
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = None

# Load environment variables
load_dotenv(r'C:\Users\axel.grille\Documents\rules-engine-agent\Agent\.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Set logger 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Dependencies loaded successfully.")
logging.info("Environment variables loaded successfully.")

# Initialize clients - Fixed OpenAI client initialization
from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

logging.info("OpenAI and Qdrant clients initialized successfully.")

# DICO API call
class DicoAPI:
    def __init__(self, base_url=None, customer=None):
        self.base_url = base_url or config.efficy_base_url
        self.customer = customer or config.efficy_customer

    def fetch_database_schema(self):
        session = requests.Session() 

        try:
            login_response = session.post( 
                f"{self.base_url}/crm/logon",
                headers={
'X-Efficy-Customer': self.customer,
                    'X-Requested-By': 'User',
                    'X-Requested-With': 'XMLHttpRequest',
                    'Content-Type': 'application/x-www-form-urlencoded'
},
                data=f'user={config.efficy_username}&password={config.efficy_password}'
            )

            if login_response.status_code == 200:
                logging.info("DICO's login successful.")

                dico_response = session.get(
                    f"{self.base_url}/crm/system/dico",
                    headers={
                        'X-Requested-By': 'User',
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                )

                if dico_response.status_code == 200:
                    logging.info("DICO data retrieved successfully.")
                    return dico_response.json()
                else: 
                    logging.error(f"Failed to retrieve DICO data: {dico_response.status_code} - {dico_response.text}")

            else: 
                logging.error(f"Login failed: {login_response.status_code} - {login_response.text}")

        except Exception as e:
            logging.error(f"An error occurred during DICO API interaction: {str(e)}")
        return None
    
class Indexer: 
    def __init__(self):
        pass

    @staticmethod
    def stable_id(*parts, length=32):
        base = '|'.join(str(p) for p in parts)
        return hashlib.sha256(base.encode()).hexdigest()[:length]

    
class UniversalTranslator:
    """Universal translator that normalizes all languages to English for better semantic matching"""
    
    def __init__(self, openai_client=None):
        self.client = openai_client or OpenAI()
        self.translation_cache = {}  # Cache translations to avoid repeated API calls
        
        # Common technical terms that shouldn't be translated
        self.technical_terms = {
            'id', 'key', 'ref', 'code', 'type', 'status', 'url', 'api', 'json', 'xml', 
            'csv', 'sql', 'uuid', 'guid', 'timestamp', 'datetime', 'boolean', 'integer',
            'varchar', 'text', 'blob', 'index', 'primary', 'foreign', 'null', 'auto'
        }
        
    def translate_column_names(self, columns: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Translate column names to English and return both translated names and mapping
        
        Args:
            columns: List of column names in any language
            
        Returns:
            Tuple of (translated_columns, translation_mapping)
        """
        if not columns:
            return [], {}
            
        # Filter out columns that are already in English or are technical terms
        columns_to_translate = []
        english_columns = []
        translation_mapping = {}
        
        for col in columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            
            # Check if column is already English (contains only English words)
            if self._is_english_column(col):
                english_columns.append(col)
                translation_mapping[col] = col  # No translation needed
            else:
                columns_to_translate.append(col)
        
        # Batch translate non-English columns
        if columns_to_translate:
            batch_translations = self._batch_translate_columns(columns_to_translate)
            translation_mapping.update(batch_translations)
            english_columns.extend(batch_translations.values())
        
        return english_columns, translation_mapping
    
    def _is_english_column(self, column: str) -> bool:
        """Check if a column name is already in English"""
        col_clean = column.lower().replace('_', '').replace('-', '').replace(' ', '')
        
        # If it's a technical term, consider it English
        if col_clean in self.technical_terms:
            return True
            
        # Check for common English patterns
        english_patterns = [
            r'^[a-z_\-\s0-9]+$',  # Only English letters, numbers, underscores, hyphens, spaces
        ]
        
        # Common English words in database columns
        english_indicators = [
            'name', 'email', 'phone', 'address', 'date', 'time', 'created', 'updated',
            'first', 'last', 'user', 'customer', 'order', 'product', 'price', 'total',
            'description', 'title', 'category', 'status', 'active', 'deleted', 'count',
            'amount', 'value', 'number', 'code', 'reference', 'contact', 'company'
        ]
        
        # If column contains English indicators, likely English
        col_words = re.findall(r'\b\w+\b', column.lower())
        if any(word in english_indicators for word in col_words):
            return True
            
        # French/other language indicators (if found, not English)
        non_english_indicators = [
            'expéditeur', 'destinataire', 'société', 'objet', 'corps', 'message',
            'saisie', 'auteur', 'état', 'suivi', 'catégorie', 'diffusion', 'priorité',
            'référence', 'campagne', 'lieu', 'désignation', 'programmé', 'fin'
        ]
        
        if any(indicator in column.lower() for indicator in non_english_indicators):
            return False
            
        # Default to English if unsure (conservative approach)
        return True
    
    def _batch_translate_columns(self, columns: List[str]) -> Dict[str, str]:
        """Translate multiple columns in a single API call for efficiency"""
        if not columns:
            return {}
            
        # Check cache first
        cached_results = {}
        columns_to_translate = []
        
        for col in columns:
            if col in self.translation_cache:
                cached_results[col] = self.translation_cache[col]
            else:
                columns_to_translate.append(col)
        
        if not columns_to_translate:
            return cached_results
            
        # Prepare batch translation prompt
        columns_text = '\n'.join([f"- {col}" for col in columns_to_translate])
        
        prompt = f"""Translate the following database column names to English. These are field names from a business database.

Rules:
1. Translate to clear, descriptive English database field names
2. Use standard database naming conventions (lowercase, underscores)
3. Keep technical terms (ID, REF, CODE, etc.) as-is
4. Make translations specific and business-appropriate
5. Return ONLY the translated names, one per line, in the same order

Column names to translate:
{columns_text}

English translations:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert database architect who translates field names to standard English database conventions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            translated_lines = response.choices[0].message.content.strip().split('\n')
            
            # Parse results and create mapping
            translations = {}
            for i, col in enumerate(columns_to_translate):
                if i < len(translated_lines):
                    translated = translated_lines[i].strip().replace('- ', '').replace('* ', '')
                    if translated:
                        translations[col] = translated
                        self.translation_cache[col] = translated  # Cache for future use
                    else:
                        translations[col] = col  # Fallback to original
                else:
                    translations[col] = col  # Fallback to original
            
            # Combine with cached results
            translations.update(cached_results)
            return translations
            
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            # Fallback: return original columns
            fallback_translations = {col: col for col in columns_to_translate}
            fallback_translations.update(cached_results)
            return fallback_translations
    
    def translate_domain_context(self, text: str) -> str:
        """Translate domain context text to English"""
        if not text or self._is_mostly_english(text):
            return text
            
        prompt = f"""Translate the following business context to clear English:

{text}

English translation:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a business translator. Translate to clear, professional English."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Context translation failed: {e}")
            return text  # Fallback to original
    
    def _is_mostly_english(self, text: str) -> bool:
        """Check if text is mostly English"""
        # Simple heuristic - if no obvious non-English characters/words
        non_english_indicators = ['é', 'è', 'ê', 'ë', 'à', 'â', 'ç', 'ù', 'û', 'ô', 'î', 'ï']
        return not any(indicator in text.lower() for indicator in non_english_indicators)


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


class GenericFileIngestionRAGPipeline:
    """
    Generic RAG pipeline with Entity-First architecture for analyzing any data file 
    and finding the best database tables using two-stage semantic search.
    """
    
    def __init__(self, qdrant_client, embeddings, collection_name, query_only=True):
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.llm_client = OpenAI()
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.txt', '.tsv']
        self.query_only = query_only  # Force query-only mode
        
        # Initialize memory for entity-relation storage
        self.memory = EntityRelationMemory()
        
        # Initialize universal translator for multilingual support
        self.translator = UniversalTranslator(self.llm_client)
        
        # Initialize embedding cache for better performance
        self.cached_embeddings = EmbeddingCache(embeddings)
        
        # Verify collection exists in query-only mode
        if self.query_only:
            existing_collections = [col.name for col in self.qdrant_client.get_collections().collections]
            if self.collection_name not in existing_collections:
                raise ValueError(f"Collection '{self.collection_name}' does not exist. Please run the RAG system in 'feed' mode first to populate the vector store.")
            logging.info(f"Query-only mode: Using existing collection '{self.collection_name}'")
    
    def analyze_file_structure(self, file_path):
        """Analyze any supported file structure and content"""
        try:
            if not os.path.exists(file_path):
                return {'error': f'File not found: {file_path}'}
            
            file_extension = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)
            
            if file_extension not in self.supported_formats:
                return {'error': f'Unsupported file format: {file_extension}'}
            
            # Handle different file types
            if file_extension == '.csv':
                return self._analyze_csv(file_path, file_name)
            elif file_extension in ['.xlsx', '.xls']:
                return self._analyze_excel(file_path, file_name)
            elif file_extension == '.json':
                return self._analyze_json(file_path, file_name)
            elif file_extension in ['.txt', '.tsv']:
                return self._analyze_text(file_path, file_name)
            
        except Exception as e:
            return {'error': f'Failed to analyze file: {str(e)}'}
    
    def _analyze_csv(self, file_path, file_name):
        """Analyze CSV files"""
        # Add delimiter detection for semicolon-separated files
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        
        delimiter = '\t' if '\t' in first_line else ',' if ',' in first_line else ';'
        df = pd.read_csv(file_path, delimiter=delimiter)
        return self._create_file_analysis(df, file_name, 'CSV')
    
    def _analyze_excel(self, file_path, file_name):
        """Analyze Excel files"""
        df = pd.read_excel(file_path)
        return self._create_file_analysis(df, file_name, 'Excel')
    
    def _analyze_json(self, file_path, file_name):
        """Analyze JSON files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            return self._create_file_analysis(df, file_name, 'JSON Array')
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
            return self._create_file_analysis(df, file_name, 'JSON Object')
        else:
            return {'error': 'JSON structure not suitable for tabular analysis'}
    
    def _analyze_text(self, file_path, file_name):
        """Analyze text/TSV files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        
        delimiter = '\t' if '\t' in first_line else ',' if ',' in first_line else ';'
        df = pd.read_csv(file_path, delimiter=delimiter)
        return self._create_file_analysis(df, file_name, 'Text/TSV')
    
    def _create_file_analysis(self, df, file_name, file_type):
        """Create standardized file analysis from DataFrame with automatic translation"""
        original_columns = df.columns.tolist()
        
        # Translate column names to English for better semantic matching
        english_columns, translation_mapping = self.translator.translate_column_names(original_columns)
        
        file_info = {
            'file_name': file_name,
            'file_type': file_type,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': original_columns,  # Keep original for user reference
            'columns_english': english_columns,  # Translated for semantic search
            'translation_mapping': translation_mapping,  # Show how columns were mapped
            'sample_data': df.head(2).to_dict('records') if len(df) > 0 else []
        }
        
        # Analyze column types and content using original columns
        column_analysis = {}
        for i, col in enumerate(original_columns):
            english_col = english_columns[i] if i < len(english_columns) else col
            column_analysis[col] = {
                'dtype': str(df[col].dtype),
                'english_name': english_col,
                'translation_used': translation_mapping.get(col, col) != col,
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
                'unique_values': df[col].nunique(),
                'sample_values': df[col].dropna().head(3).tolist()
            }
        
        file_info['column_analysis'] = column_analysis
        
        # Log translation results for debugging
        translated_count = sum(1 for original, translated in translation_mapping.items() 
                              if original != translated)
        if translated_count > 0:
            logging.info(f"Translated {translated_count}/{len(original_columns)} columns to English")
            logging.info(f"Sample translations: {dict(list(translation_mapping.items())[:3])}")
        
        return file_info
    
    def rank_tables_by_relevance(self, search_results):
        """Rank tables by multiple relevance criteria"""
        ranked_tables = []
        
        for table_name, data in search_results.items():
            # Handle both search methods' return formats
            if 'scores' in data:
                # Multi-level search format
                scores = data['scores']
                avg_score = sum(scores) / len(scores) if scores else 0
                max_score = max(scores) if scores else 0
                query_coverage = len(set(data['queries_matched']))
                # Normalize query_coverage to 0-1 range (assume max 10 queries)
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
                # Normalize query_coverage to 0-1 range (assume max 10 queries)
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
                'queries_matched': data['queries_matched']
            })
        
        ranked_tables.sort(key=lambda x: x['composite_score'], reverse=True)
        return ranked_tables

    def run_complete_pipeline(self, file_path, user_context=None):
        """Run the complete RAG pipeline for any file type"""
        logging.info("=== GENERIC FILE INGESTION RAG PIPELINE ===")
        logging.info(f"Analyzing: {os.path.basename(file_path)}")
                
        # Step 1: Analyze file structure
        logging.info("Step 1: Analyzing file structure...")
        file_info = self.analyze_file_structure(file_path)
        if 'error' in file_info:
            return file_info
        
        logging.info(f"{file_info['file_type']} file: {file_info['total_columns']} columns, {file_info['total_rows']} rows")
        logging.info(f"Detected domain: {self._infer_data_domain(file_info['columns'])['primary_domain']}")
                
        # Step 2: Generate semantic search queries
        logging.info("Step 2: Generating semantic search queries...")
        queries = self.generate_semantic_queries(file_info, user_context)
        logging.info(f"Generated {len(queries)} queries for database search")
                
        # Step 3: Search for relevant tables
        logging.info("Step 3: Searching for relevant database tables...")
        search_results = self.search_relevant_tables(queries)
        logging.info(f"Found {len(search_results)} unique tables across all queries")
                
        # Step 4: Rank tables by relevance
        logging.info("Step 4: Ranking tables by relevance...")
        ranked_tables = self.rank_tables_by_relevance(search_results)
        logging.info(f"Ranked {len(ranked_tables)} tables by composite relevance score")
                
        # Compile final results
        final_results = {
            'file_analysis': file_info,
            'inferred_domain': self._infer_data_domain(file_info['columns']),
            'user_context': user_context,
            'search_queries_used': queries,
            'total_tables_found': len(search_results),
            'top_10_tables': ranked_tables[:10],
            'ingestion_summary': {
                'recommended_table': ranked_tables[0]['table_name'] if ranked_tables else None,
                'confidence_level': self._calculate_confidence_level(ranked_tables[0] if ranked_tables else None),
                'requires_review': self._requires_review(ranked_tables[0] if ranked_tables else None),
                'mapping_ready': len(ranked_tables) > 0 and ranked_tables[0]['composite_score'] > 0.6
            }
        }
        
        return final_results

    def _calculate_confidence_level(self, best_table):
        """Calculate confidence level for SQL agent"""
        if not best_table:
            return 'None'
        score = best_table['composite_score']
        if score > 0.6:  # Lowered from 0.8
            return 'High'
        elif score > 0.4:  # Lowered from 0.6
            return 'Medium'
        else:
            return 'Low'
    
    def _requires_review(self, best_table):
        """Determine if human review is needed before SQL generation"""
        if not best_table:
            return True
        return best_table['composite_score'] < 0.7
    
    def _calculate_confidence_level_vector(self, best_table):
        """Calculate confidence level based on vector similarity score"""
        if not best_table:
            return 'None'
        score = best_table['vector_score']
        if score > 0.8:
            return 'High'
        elif score > 0.65:
            return 'Medium'
        else:
            return 'Low'
    
    def _requires_review_vector(self, best_table):
        """Determine if human review is needed based on vector score"""
        if not best_table:
            return True
        return best_table['vector_score'] < 0.7

    def display_results_summary(self, results):
        """Display a formatted summary optimized for SQL agent consumption"""
        if 'error' in results:
            logging.error(f"Pipeline Error: {results['error']}")
            return
        
        file_info = results['file_analysis']
        summary = results['ingestion_summary']
        
        logging.info("=" * 80)
        logging.info(f"INGESTION ANALYSIS: {file_info['file_name']}")
        logging.info("=" * 80)
        logging.info(f"File: {file_info['file_type']} | {file_info['total_rows']} rows | {file_info['total_columns']} columns")
        logging.info(f"Domain: {results['inferred_domain']['primary_domain']}")
        logging.info(f"Best Table: {summary['recommended_table']}")
        logging.info(f"Confidence: {summary['confidence_level']}")
        logging.info(f"Review Required: {'Yes' if summary['requires_review'] else 'No'}")
        logging.info(f"Mapping Ready: {'Yes' if summary['mapping_ready'] else 'No'}")
        
        logging.info("TOP 10 DATABASE TABLES:") 
        for i, table in enumerate(results['top_10_tables'], 1):  
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
            output_file = rf"results_for_agent\{base_name}_ingestion_analysis.json"
        
        sql_agent_data = {
            'source_file': results['file_analysis']['file_name'],
            'file_structure': {
                'columns': results['file_analysis']['columns'],
                'total_rows': results['file_analysis']['total_rows'],
                'column_types': {col: analysis['dtype'] for col, analysis in results['file_analysis']['column_analysis'].items()}
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
                for table in results['top_10_tables']
            ],
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sql_agent_data, f, indent=2, ensure_ascii=False)
            logging.info(f"SQL Agent data exported to: {output_file}")
            return {'success': True, 'output_file': output_file}
        except Exception as e:
            return {'error': f'Export failed: {str(e)}'}

    def _infer_data_domain(self, columns):
        "Enhanced data domain inference with multilingual support and comprehensive business entity detection"
        columns_lower = [col.lower() for col in columns]
        
        # Multilingual domain detection patterns (English + French + other languages)
        domain_patterns = {
            # Core CRM entities - multilingual
            'leads': [
                'lead', 'prospect', 'lead_status', 'source', 'qualification', 'score', 'conversion',
                'lead_id', 'prospect_id', 'qualified', 'unqualified', 'mql', 'sql', 'nurture',
                'prospectus', 'candidat', 'piste', 'qualification_prospect'
            ],
            'opportunities': [
                'opportunity', 'deal', 'pipeline', 'stage', 'probability', 'close_date', 'forecast',
                'opp_id', 'deal_id', 'sales_stage', 'win_probability', 'expected_revenue', 'deal_value',
                'opportunité', 'affaire', 'vente', 'étape', 'probabilité', 'revenus'
            ],
            'contacts': [
                'contact', 'person', 'individual', 'first_name', 'last_name', 'title', 'relationship',
                'contact_id', 'person_id', 'full_name', 'job_title', 'phone', 'mobile', 'email',
                'personne', 'individu', 'prénom', 'nom', 'nom_famille', 'titre', 'téléphone', 'courriel'
            ],
            'companies': [
                'company', 'organization', 'enterprise', 'business', 'industry', 'sector', 'headquarters',
                'company_id', 'org_id', 'business_name', 'company_name', 'industry_type', 'company_size',
                'société', 'entreprise', 'organisation', 'industrie', 'secteur', 'siège'
            ],
            'activities': [
                'activity', 'action', 'event', 'log', 'history', 'timeline', 'interaction',
                'activity_id', 'event_id', 'action_type', 'activity_type', 'interaction_type', 'follow_up',
                'activité', 'événement', 'historique', 'interaction', 'suivi', 'visite'
            ],
            'meetings': [
                'meeting', 'appointment', 'schedule', 'calendar', 'attendee', 'agenda', 'duration',
                'meeting_id', 'appointment_id', 'scheduled', 'start_time', 'end_time', 'location',
                'réunion', 'rendez-vous', 'calendrier', 'participant', 'durée', 'lieu'
            ],
            'campaigns': [
                'campaign', 'marketing', 'promotion', 'advertising', 'channel', 'target', 'response',
                'campaign_id', 'promo_id', 'marketing_campaign', 'campaign_name', 'campaign_type',
                'campagne', 'promotion', 'publicité', 'canal', 'cible', 'réponse', 'diffusion'
            ],
            'tickets': [
                'ticket', 'issue', 'support', 'incident', 'priority', 'resolution', 'escalation',
                'ticket_id', 'issue_id', 'support_ticket', 'incident_id', 'case_id', 'help_desk',
                'billet', 'problème', 'incident', 'priorité', 'résolution', 'escalade'
            ],
            'users': [
                'user', 'username', 'login', 'profile', 'role', 'permission', 'access',
                'user_id', 'account', 'user_name', 'login_name', 'user_role', 'access_level',
                'utilisateur', 'profil', 'rôle', 'permission', 'accès', 'compte'
            ],
            'communication': [
                'mail', 'email', 'message', 'subject', 'body', 'sender', 'recipient', 'cc', 'bcc',
                'expéditeur', 'destinataire', 'objet', 'corps', 'canal', 'catégorie', 'diffusion',
                'visite', 'mail_cc', 'mail_expéditeur', 'mail_destinataire', 'programmé'
            ],
            'sales_orders': [
                'order', 'product', 'price', 'quantity', 'total', 'invoice', 'payment',
                'commande', 'produit', 'prix', 'quantité', 'total', 'facture', 'paiement'
            ],
            'financial': [
                'amount', 'cost', 'revenue', 'budget', 'transaction', 'account', 'currency',
                'montant', 'coût', 'revenus', 'budget', 'transaction', 'compte', 'devise'
            ],
            'hr_employee': [
                'employee', 'staff', 'salary', 'department', 'position', 'hire', 'manager',
                'employé', 'personnel', 'salaire', 'département', 'poste', 'embauche', 'manager'
            ],
            'inventory': [
                'item', 'stock', 'warehouse', 'supplier', 'category', 'sku', 'unit',
                'article', 'stock', 'entrepôt', 'fournisseur', 'catégorie', 'unité'
            ],
            'project': [
                'project', 'task', 'milestone', 'deadline', 'status', 'resource', 'team',
                'projet', 'tâche', 'jalon', 'échéance', 'statut', 'ressource', 'équipe'
            ],
            'logistics': [
                'shipment', 'delivery', 'tracking', 'carrier', 'destination', 'weight',
                'expédition', 'livraison', 'suivi', 'transporteur', 'destination', 'poids'
            ]
        }
        
        # Calculate domain scores with weighted importance and multilingual matching
        domain_scores = {}
        for domain, keywords in domain_patterns.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                for col in columns_lower:
                    # Exact match (highest priority)
                    if keyword == col:
                        score += 5
                        matched_keywords.append(keyword)
                    # Substring match 
                    elif keyword in col:
                        # Special boost for French email terms in communication domain
                        if domain == 'communication' and keyword in [
                            'mail expéditeur', 'mail destinataire', 'corps du message', 'objet',
                            'expéditeur', 'destinataire', 'canal', 'catégorie', 'diffusion'
                        ]:
                            score += 4  
                        # Weight ID fields higher (strong indicators)
                        elif keyword.endswith('_id') or keyword.endswith('_ref'):
                            score += 3
                        # Regular substring matches
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
        
        # Enhanced domain mapping with confidence indicators
        domain_mapping = {
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
            'meetings': {
                'primary_domain': 'meeting and calendar management', 
                'business_area': 'scheduling and appointments', 
                'data_category': 'meetings',
                'table_hints': ['meeting', 'appointment', 'schedule']
            },
            'campaigns': {
                'primary_domain': 'marketing campaign management', 
                'business_area': 'marketing operations', 
                'data_category': 'campaigns',
                'table_hints': ['campaign', 'marketing', 'promotion']
            },
            'tickets': {
                'primary_domain': 'ticketing and support management', 
                'business_area': 'customer support', 
                'data_category': 'tickets',
                'table_hints': ['ticket', 'support', 'incident']
            },
            'users': {
                'primary_domain': 'user and account management', 
                'business_area': 'system administration', 
                'data_category': 'users',
                'table_hints': ['user', 'account', 'profile']
            },
            'communication': {
                'primary_domain': 'communication and messaging', 
                'business_area': 'correspondence', 
                'data_category': 'communication',
                'table_hints': ['mail', 'email', 'message', 'visit']
            },
            'sales_orders': {'primary_domain': 'sales and order management', 'business_area': 'sales operations', 'data_category': 'transactional', 'table_hints': ['order', 'sale', 'invoice']},
            'financial': {'primary_domain': 'financial and accounting', 'business_area': 'finance', 'data_category': 'financial', 'table_hints': ['financial', 'accounting', 'budget']},
            'hr_employee': {'primary_domain': 'human resources', 'business_area': 'HR management', 'data_category': 'employee', 'table_hints': ['employee', 'staff', 'hr']},
            'inventory': {'primary_domain': 'inventory and stock management', 'business_area': 'supply chain', 'data_category': 'inventory', 'table_hints': ['inventory', 'stock', 'product']},
            'project': {'primary_domain': 'project management', 'business_area': 'project operations', 'data_category': 'project', 'table_hints': ['project', 'task', 'milestone']},
            'logistics': {'primary_domain': 'logistics and shipping', 'business_area': 'operations', 'data_category': 'logistics', 'table_hints': ['shipment', 'delivery', 'logistics']},
            'general': {'primary_domain': 'business data', 'business_area': 'general operations', 'data_category': 'business', 'table_hints': ['data', 'general']}
        }
        
        result = domain_mapping.get(best_domain, domain_mapping['general'])
        
        # Add detection metadata for debugging and confidence assessment
        result['detection_confidence'] = min(best_score / 5.0, 1.0)  # Normalize to 0-1
        result['matched_keywords'] = domain_scores.get(best_domain, {}).get('matched_keywords', [])
        result['all_scores'] = {k: v['score'] for k, v in domain_scores.items() if v['score'] > 0}
        
        return result
    
    def generate_semantic_queries(self, file_info, user_context=None):
        """Generate semantic queries based on file content and optional user context"""
        # Use English columns for domain detection (better semantic matching)
        columns_for_analysis = file_info.get('columns_english', file_info.get('columns', []))
        domain_hints = self._infer_data_domain(columns_for_analysis)
        file_name = file_info.get('file_name', 'unknown')
        file_type = file_info.get('file_type', 'unknown')
        
        # Translate user context if needed
        if user_context:
            user_context = self.translator.translate_domain_context(user_context)
        
        # Enhanced query templates using English column names for better matching
        query_templates = [
            f"database table for storing {domain_hints['primary_domain']} data with fields like {', '.join(columns_for_analysis[:6])}",
            f"{file_type} data ingestion into relational database with columns {', '.join(columns_for_analysis[:8])}",
            f"business data management system for {domain_hints['business_area']} information",
            f"data warehouse table for {domain_hints['data_category']} records and analytics",
            f"structured data storage for {file_name} content in enterprise database"
        ]

        # Add domain-specific targeted queries with enhanced multilingual support
        if domain_hints['data_category'] == 'communication':
            query_templates.extend([
                "email communication tracking and management system with sender recipient subject message",
                "mail message storage organization with expéditeur destinataire objet corps du message",
                "correspondence interaction history email sent received tracking with french fields",
                "visit mail communication log with sender recipient date subject priority canal",
                "business email workflow tracking system with mail status categories and diffusion",
                "customer communication management with email threading follow-up visite_mail_cc",
                "mail interaction database with french email fields mail expéditeur destinataire",
                "email correspondence tracking with canal catégorie diffusion programmé priorité",
                "visite mail tracking system with cl_ref société nom date saisie auteur",
                "email management with french corporate fields société nom objet type canal",
                "correspondence tracking with état date de suivi référence campagne lieu"
            ])
        elif domain_hints['data_category'] in ['contacts', 'companies']:
            query_templates.extend([
                "contact person individual management with names emails phones addresses",
                "company organization enterprise business with industry sector headquarters",
                "customer relationship management contact information database",
                "business partner contact details and communication preferences"
            ])
        elif domain_hints['data_category'] in ['activities', 'meetings']:
            query_templates.extend([
                "activity event interaction tracking with dates types participants",
                "meeting appointment calendar scheduling with attendees duration location",
                "business interaction history timeline and follow-up activities",
                "customer visit activity log with outcomes and next steps"
            ])
        else:
            query_templates.extend([
                "email communication tracking and management system",
                "customer correspondence and interaction history", 
                "mail message storage and organization",
                "contact information and relationship management",
                "document and file management system",
                "business communication workflow and tracking"
            ])

        
        if user_context:
            context_query = f"{user_context} with data structure: {', '.join(columns_for_analysis[:10])}"
            query_templates.insert(0, context_query)
        
        return query_templates
    
    def generate_user_context_by_file_type(self, file_path):
        """Generate user context prompt based on file type and columns with proper error handling"""
        try:
            # Check file size to avoid reading huge files
            file_size = os.path.getsize(file_path)
            max_file_size = 10 * 1024 * 1024  # 10MB limit
            
            if file_size > max_file_size:
                logging.warning(f"File too large ({file_size} bytes), using filename and structure analysis only")
                file_info = self.analyze_file_structure(file_path)
                if 'error' in file_info:
                    return "Importing data from the file into the CRM system by mapping fields to the most relevant database table."
                
                columns = file_info.get('columns', [])
                file_name = os.path.basename(file_path)
                domain_info = self._infer_data_domain(columns)
                
                return f"The file appears to contain {domain_info['primary_domain']} data. Provide context for data ingestion considering columns: {', '.join(columns[:10])}. Find the best matching database tables and their relationships in the existing database schema."
            
            # Get file analysis (reuse existing analysis if available)
            file_info = self.analyze_file_structure(file_path)
            if 'error' in file_info:
                logging.error(f"Failed to analyze file structure: {file_info['error']}")
                return "Importing data from the file into the CRM system by mapping fields to the most relevant database table."
            
            columns = file_info.get('columns', [])
            file_name = os.path.basename(file_path)
            sample_data = file_info.get('sample_data', [])
            
            # Create a concise content summary instead of reading entire file
            content_summary = f"File: {file_name}\n"
            content_summary += f"Columns: {', '.join(columns)}\n"
            if sample_data:
                content_summary += f"Sample data (first 2 rows): {sample_data}\n"
            content_summary += f"Total rows: {file_info.get('total_rows', 'unknown')}\n"
            content_summary += f"File type: {file_info.get('file_type', 'unknown')}"
            
            # Use OpenAI to classify the domain
            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert data analyst specialized in identifying business domains from file content. Respond with only the primary domain as a single word or short phrase (e.g., email, appointment, lead, opportunity, contact, company, activity, meeting, campaign, ticket, user, sales_order, financial, hr_employee, inventory, project, logistics)."
                        },
                        {
                            "role": "user", 
                            "content": f"Analyze the following file structure and determine the most relevant business domain:\n\n{content_summary}\n\nProvide only the primary domain."
                        }
                    ],
                    max_tokens=50,
                    temperature=0.0
                )
                
                domain = response.choices[0].message.content.strip().lower()
                logging.info(f"LLM classified domain as: {domain}")
                
            except Exception as e:
                logging.error(f"Failed to classify domain with LLM: {e}")
                # Fallback to existing domain inference
                domain_info = self._infer_data_domain(columns)
                domain = domain_info['data_category']
                logging.info(f"Using fallback domain classification: {domain}")
            
            # Generate context prompt
            prompt = f"The file is related to {domain}. Find the best matching database tables and their relationships in the existing database schema for storing this {domain} data."
            
            logging.info(f"Generated user context: {prompt[:100]}...")
            return prompt
            
        except Exception as e:
            logging.error(f"Error in generate_user_context_by_file_type: {e}")
            return "Importing data from the file into the CRM system by mapping fields to the most relevant database table."

    def search_relevant_tables(self, queries, top_k=10):
        """Search for relevant tables using multiple semantic queries"""
        all_results = {}
        
        for i, query in enumerate(queries):
            logging.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                query_embedding = self.cached_embeddings.embed_query(query)
                
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

    def search_relevant_tables_multi_level(self, queries, top_k=10):
        """Enhanced search across different chunk types with weighted scoring"""
        all_results = {}
        
        # Define chunk type priorities and weights
        chunk_priorities = {
            "table_summary": 1.0,
            "field_group": 0.8,
            "table_ingestion_profile": 0.9,
            "table_ingestion_profile_complete": 0.6,
            "relationship_focus": 0.4
        }
        
        for i, query in enumerate(queries):
            logging.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            for chunk_type, weight in chunk_priorities.items():
                try:
                    query_embedding = self.cached_embeddings.embed_query(query)
                    
                    search_results = self.qdrant_client.query_points(
                        collection_name=self.collection_name,
                        query=query_embedding,
                        query_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="chunk_type",
                                    match=models.MatchValue(value=chunk_type)
                                )
                            ]
                        ),
                        limit=top_k
                    )
                    
                    for point in search_results.points:
                        table_name = point.payload['primary_table']
                        weighted_score = point.score * weight
                        
                        if table_name not in all_results:
                            all_results[table_name] = {
                                'table_name': table_name,
                                'table_code': point.payload['table_code'],
                                'table_kind': point.payload.get('table_kind', 'Entity'),
                                'field_count': point.payload.get('field_count', 0),
                                'scores': [],
                                'chunk_types': [],
                                'best_chunks': {},
                                'queries_matched': []
                            }
                        
                        all_results[table_name]['scores'].append(weighted_score)
                        all_results[table_name]['chunk_types'].append(chunk_type)
                        all_results[table_name]['queries_matched'].append(i+1)
                        
                        # Keep best chunk of each type
                        if chunk_type not in all_results[table_name]['best_chunks'] or \
                           weighted_score > all_results[table_name]['best_chunks'][chunk_type]['score']:
                            all_results[table_name]['best_chunks'][chunk_type] = {
                                'content': point.payload.get('content', ''),
                                'score': weighted_score,
                                'metadata': point.payload.get('metadata', {})
                            }
                            
                except Exception as e:
                    logging.error(f"Search error for query {i+1}, chunk_type {chunk_type}: {e}")
        
        return all_results

    def _detect_relationship_data(self, file_info):
        """Detect if source data appears to be relationship/mapping data"""
        columns = file_info.get('columns', [])
        columns_lower = [col.lower() for col in columns]
        
        # Indicators of relationship data
        relationship_indicators = [
            # Direct relationship indicators
            'mapping', 'relation', 'link', 'association', 'junction',
            # Foreign key patterns
            '_id', 'id_', 'ref_', '_ref', 'key_', '_key', 'K',
            # Many-to-many patterns
            'user_role', 'contact_company', 'product_category', 'document_folder',
            # Lookup/reference patterns
            'lookup', 'reference', 'xref', 'cross_ref'
        ]
        
        relationship_score = 0
        total_columns = len(columns)
        
        # Check for relationship patterns
        for col in columns_lower:
            for indicator in relationship_indicators:
                if indicator in col:
                    relationship_score += 1
                    break
        
        # Additional checks
        # High ratio of ID/reference fields
        id_columns = sum(1 for col in columns_lower if col.endswith('_id') or col.startswith('id_') or 'ref' in col)
        id_ratio = id_columns / max(total_columns, 1)
        
        # Few descriptive fields (mostly IDs and references)
        descriptive_fields = ['name', 'title', 'description', 'email', 'phone', 'address', 'date']
        descriptive_count = sum(1 for col in columns_lower for field in descriptive_fields if field in col)
        
        # Decision logic
        is_relationship_data = (
            (relationship_score >= 2) or  # Multiple relationship indicators
            (id_ratio > 0.5 and total_columns <= 5) or  # Many IDs, few columns
            (id_ratio > 0.7) or  # Very high ID ratio
            (descriptive_count == 0 and total_columns <= 4)  # No descriptive fields, few columns
        )
        
        logging.info(f"Relationship detection - Score: {relationship_score}, ID ratio: {id_ratio:.2f}, "
                    f"Descriptive fields: {descriptive_count}, Decision: {is_relationship_data}")
        
        return is_relationship_data

    def run_entity_first_pipeline(self, file_path, user_context=None):
        """Run the complete Entity-First two-stage pipeline"""
        logging.info("=== ENTITY-FIRST RAG PIPELINE ===")
        logging.info(f"Analyzing: {os.path.basename(file_path)}")
        
        # Step 1: Analyze file structure
        logging.info("Step 1: Analyzing file structure...")
        file_info = self.analyze_file_structure(file_path)
        if 'error' in file_info:
            return file_info
        
        logging.info(f"{file_info['file_type']} file: {file_info['total_columns']} columns, {file_info['total_rows']} rows")
        
        # Step 2: Detect if this is relationship data
        is_relationship_data = self._detect_relationship_data(file_info)
        if is_relationship_data:
            self.memory.set_relationship_data_flag(True)
            logging.info("🔗 Detected potential relationship/mapping data")
        
        # Step 3: Generate semantic search queries
        logging.info("Step 3: Generating semantic search queries...")
        queries = self.generate_semantic_queries(file_info, user_context)
        logging.info(f"Generated {len(queries)} queries for database search")
        
        # STAGE 1: Entity-First Search
        logging.info("\n=== STAGE 1: ENTITY-FIRST SEARCH ===")
        entity_results = self.search_entity_tables_only(queries)
        logging.info(f"Found {len(entity_results)} entity tables")
        
        # Rank entity results
        ranked_entities = self.rank_tables_by_relevance(entity_results)
        
        # Store entities in memory with higher confidence threshold
        entities_stored = 0
        for entity in ranked_entities:
            if entity['composite_score'] >= self.memory.confidence_threshold_entity:
                self.memory.store_entity(entity['table_name'], entity, entity['composite_score'])
                entities_stored += 1
        
        logging.info(f"Stored {entities_stored} high-confidence entities (threshold: {self.memory.confidence_threshold_entity})")
        
        # STAGE 2: Relationship Discovery
        logging.info("\n=== STAGE 2: RELATIONSHIP DISCOVERY ===")
        
        if self.memory.relationship_data_flag:
            # Search for relation tables if relationship data detected
            relation_results = self.search_relation_tables_only(queries)
            logging.info(f"Found {len(relation_results)} relation tables for relationship data")
        else:
            # Search for related tables based on best entity
            best_entity = self.memory.get_best_entity()
            if best_entity:
                relation_results = self.search_related_tables(queries, best_entity['table_name'])
                logging.info(f"Found {len(relation_results)} tables related to {best_entity['table_name']}")
            else:
                relation_results = {}
                logging.warning("No high-confidence entities found, skipping relationship discovery")
        
        # Rank and store relation results
        if relation_results:
            ranked_relations = self.rank_tables_by_relevance(relation_results)
            
            relations_stored = 0
            for relation in ranked_relations:
                if relation['composite_score'] >= self.memory.confidence_threshold_relation:
                    # Determine related entities for this relation
                    related_entities = self._find_related_entities(relation['table_name'])
                    self.memory.store_relation(relation['table_name'], relation, 
                                             relation['composite_score'], related_entities)
                    relations_stored += 1
            
            logging.info(f"Stored {relations_stored} relations (threshold: {self.memory.confidence_threshold_relation})")
        
        # Compile final results
        memory_summary = self.memory.get_summary()
        
        final_results = {
            'pipeline_type': 'entity_first_two_stage',
            'file_analysis': file_info,
            'inferred_domain': self._infer_data_domain(file_info['columns']),
            'user_context': user_context,
            'search_queries_used': queries,
            'relationship_data_detected': memory_summary['relationship_data_detected'],
            'entities_discovered': memory_summary['entities_discovered'],
            'relations_discovered': memory_summary['relations_discovered'],
            'memory_summary': memory_summary,
            'ingestion_summary': self._create_entity_first_summary(memory_summary),
            'stage1_entity_results': ranked_entities[:5] if ranked_entities else [],
            'stage2_relation_results': ranked_relations[:5] if 'ranked_relations' in locals() else []
        }
        
        return final_results

    def search_entity_tables_only(self, queries, top_k=10):
        """Stage 1: Search only Entity tables, completely excluding Relations"""
        all_results = {}
        
        for i, query in enumerate(queries):
            logging.info(f"Entity search query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                query_embedding = self.cached_embeddings.embed_query(query)
                
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
                                match=models.MatchValue(value="Entity")  # Only Entity tables
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
                logging.error(f"Entity search error for query {i+1}: {e}")
        
        return all_results

    def search_relation_tables_only(self, queries, top_k=10):
        """Stage 2a: Search only Relation tables for relationship data"""
        all_results = {}
        
        for i, query in enumerate(queries):
            logging.info(f"Relation search query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                query_embedding = self.cached_embeddings.embed_query(query)
                
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
                                match=models.MatchValue(value="Relation")  # Only Relation tables
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
        """Stage 2b: Search for related Entity + Relation tables based on selected entity"""
        all_results = {}
        
        # Enhanced queries with entity context
        entity_enhanced_queries = queries + [
            f"tables related to {entity_table_name} entity for data relationships",
            f"junction tables connecting {entity_table_name} to other entities",
            f"foreign key relationships with {entity_table_name} table",
            f"many-to-many associations involving {entity_table_name}"
        ]
        
        for i, query in enumerate(entity_enhanced_queries):
            logging.info(f"Related search query {i+1}/{len(entity_enhanced_queries)}: {query[:50]}...")
            
            try:
                query_embedding = self.cached_embeddings.embed_query(query)
                
                # Search both Entity and Relation tables (no table_kind filter)
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
                    
                    # Skip the original entity table
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

    def _find_related_entities(self, relation_table_name):
        """Find which entities are related to a given relation table"""
        # This is a simplified implementation - could be enhanced with actual relationship analysis
        related_entities = []
        
        # Look for entities that might be connected to this relation
        for entity_name in self.memory.entities.keys():
            # Simple heuristic: if entity name appears in relation name
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
            # Relationship data found good relation tables
            best_relations = memory_summary.get('all_relations', [])
            recommended_table = best_relations[0]['table_name'] if best_relations else None
            confidence_level = self._calculate_confidence_level_from_score(best_relations[0]['confidence_score']) if best_relations else 'None'
            table_type = 'relation'
        elif best_entity:
            # Found good entity table
            recommended_table = best_entity['table_name']
            confidence_level = self._calculate_confidence_level_from_score(best_entity['confidence_score'])
            table_type = 'entity'
        else:
            # No good matches found
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
            'pipeline_success': mapping_ready or (entities_count > 0 or relations_count > 0),
            'table_classification': 'unknown_type_table' if not mapping_ready and entities_count == 0 and relations_count == 0 else table_type
        }

    def _calculate_confidence_level_from_score(self, score):
        """Calculate confidence level from composite score"""
        if score >= 0.6:  # Lowered from 0.8
            return 'High'
        elif score >= 0.4:  # Lowered from 0.6
            return 'Medium'
        else:
            return 'Low'

    # Keep the original method for backward compatibility
    def run_complete_pipeline(self, file_path, user_context=None):
        """Legacy method - redirects to entity-first pipeline"""
        logging.info("Redirecting to Entity-First pipeline...")
        return self.run_entity_first_pipeline(file_path, user_context)

logging.info("GenericFileIngestionRAGPipeline class complete with all methods")

                 


def feed_vector_store(dico_api, qdrant_client, embeddings, collection_name):
    """Feed mode: Fetch DICO data, create chunks, and populate vector store """
    logging.info("=== FEED MODE: Building Vector Store ===")
    
    # Fetch DICO data
    logging.info("Fetching database schema from DICO API...")
    dico_data = dico_api.fetch_database_schema()
    if not dico_data:
        logging.error("Failed to fetch DICO data")
        return False
    
    # Generate table chunks
    logging.info("Generating table chunks...")
    table_chunks = generate_table_ingestion_chunks(dico_data)
    logging.info(f"Generated {len(table_chunks)} table chunks for ingestion.")

    # Check/create collection
    existing_collections = [col.name for col in qdrant_client.get_collections().collections]
    if collection_name not in existing_collections: 
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(embeddings.embed_query("Hello world")),
                distance=models.Distance.COSINE,
            ),
        )
        logging.info(f"Created new collection: {collection_name}")
    else: 
        logging.info(f"Using existing collection: {collection_name}")

    # Individual embedding creation
    logging.info("Creating individual embeddings...")
    
    # Create points for vector store
    table_points = []
    for chunk in table_chunks:
        chunk_id = Indexer.stable_id(
            chunk.metadata['chunk_type'],
            chunk.metadata['primary_table'],
            chunk.metadata['table_code']
        )

        # Individual embedding creation
        embedding = embeddings.embed_query(chunk.page_content)

        point = PointStruct(
            id=chunk_id,
            vector=embedding,
            payload={
                'content': chunk.page_content,
                'chunk_type': chunk.metadata['chunk_type'],
                'primary_table': chunk.metadata['primary_table'],
                'table_code': chunk.metadata['table_code'],
                'table_kind': chunk.metadata['table_kind'],
                'field_count': chunk.metadata['field_count'],
                'metadata': chunk.metadata
            }
        )
        table_points.append(point)

    # Upsert points to vector store
    try:
        result = qdrant_client.upsert(
            collection_name=collection_name, 
            points=table_points
        )
        logging.info(f"Successfully upserted {len(table_points)} table chunks")
        collection_info = qdrant_client.get_collection(collection_name)
        logging.info(f"Collection now contains {collection_info.points_count} points.")
    except Exception as e:
        logging.error(f"Error during upsert: {str(e)}")
        return False

    # Create payload indexes for efficient filtering 
    try:
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="chunk_type",
            field_schema=PayloadSchemaType.KEYWORD
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="primary_table",
            field_schema=PayloadSchemaType.KEYWORD
        )
        logging.info('Payload indexes created successfully.')
    except Exception as e:
        logging.error(f"Error creating payload indexes: {str(e)}")
    
    logging.info("=== FEED MODE COMPLETED SUCCESSFULLY ===")
    return True


def query_vector_store(file_path, qdrant_client, embeddings, collection_name):
    """Query mode: Directly query existing vector store for file analysis"""
    logging.info("=== QUERY MODE: Analyzing File Against Existing Vector Store ===")
    
    # Check if collection exists
    existing_collections = [col.name for col in self.qdrant_client.get_collections().collections]
    if collection_name not in existing_collections:
        logging.error(f"Collection '{collection_name}' does not exist. Please run in 'feed' mode first.")
        return None
    
    # Initialize pipeline
    pipeline = GenericFileIngestionRAGPipeline(qdrant_client, embeddings, collection_name)
    logging.info("Initialized GenericFileIngestionRAGPipeline")

    # Generate user context based on file type
    user_context = pipeline.generate_user_context_by_file_type(file_path)
    
    # Run complete pipeline
    logging.info('Running complete RAG pipeline...')
    results = pipeline.run_complete_pipeline(file_path, user_context)
    pipeline.display_results_summary(results)

    # Export results for SQL agent
    if 'error' not in results:
        export_result = pipeline.export_for_sql_agent(results)
        if 'success' in export_result:
            logging.info(f'Exported SQL agent data to {export_result["output_file"]}')
    
    return results


def main():
    """Main function with mode selection"""
    import sys
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python RAG_maxo_database.py <mode> [file_path]")
        print("Modes:")
        print("  feed   - Fetch DICO data and populate vector store")
        print("  query  - Query existing vector store with a file")
        print("Examples:")
        print("  python RAG_maxo_database.py feed")
        print("  python RAG_maxo_database.py query path/to/your/file.csv")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode not in ['feed', 'query']:
        print("Error: Mode must be either 'feed' or 'query'")
        sys.exit(1)
    
    if mode == 'query' and len(sys.argv) < 3:
        print("Error: Query mode requires a file path")
        print("Usage: python RAG_maxo_database.py query <file_path>")
        sys.exit(1)
    
    # Initialize DICO API
    dico_getter = DicoAPI(
        base_url="https://sandbox-5.efficytest.cloud",
        customer="SANDBOX05"
    )
    
    # Create collection name
    collection_name = "maxo_vector_store_v2" #f"{dico_getter.customer.lower()}_database_schema"
    
    if mode == 'feed':
        success = feed_vector_store(dico_getter, qdrant_client, embeddings, collection_name)
        if success:
            print("Vector store feeding completed successfully!")
        else:
            print("Vector store feeding failed!")
            sys.exit(1)
    
    elif mode == 'query':
        file_path = sys.argv[2]
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        
        results = query_vector_store(file_path, qdrant_client, embeddings, collection_name)
        if results is None:
            print("Query failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
