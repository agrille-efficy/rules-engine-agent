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

from .chunk_generator import generate_table_ingestion_chunks
from .config import config

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

    def stable_id(*parts, length=32):
        base = '|'.join(str(p) for p in parts)
        return hashlib.sha256(base.encode()).hexdigest()[:length]

    
class GenericFileIngestionRAGPipeline:
    """
    Generic RAG pipeline for analyzing any data file and finding the top 5 best
    database tables for data ingestion using semantic search and LLM analysis.
    """
    
    def __init__(self, qdrant_client, embeddings, collection_name, query_only=True):
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.llm_client = OpenAI()
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.txt', '.tsv']
        self.query_only = query_only  # Force query-only mode
        
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
        df = pd.read_csv(file_path)
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
        """Create standardized file analysis from DataFrame"""
        file_info = {
            'file_name': file_name,
            'file_type': file_type,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'sample_data': df.head(2).to_dict('records') if len(df) > 0 else []
        }
        
        # Analyze column types and content
        column_analysis = {}
        for col in df.columns:
            column_analysis[col] = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
                'unique_values': df[col].nunique(),
                'sample_values': df[col].dropna().head(3).tolist()
            }
        
        file_info['column_analysis'] = column_analysis
        return file_info
    
    def _infer_data_domain(self, columns):
        "Enhanced data domain inference with comprehensive business entity detection"
        columns_lower = [col.lower() for col in columns]
        
        # Comprehensive domain detection patterns
        domain_patterns = {
            # Core CRM entities
            'leads': [
                'lead', 'prospect', 'lead_status', 'source', 'qualification', 'score', 'conversion',
                'lead_id', 'prospect_id', 'qualified', 'unqualified', 'mql', 'sql', 'nurture'
            ],
            'opportunities': [
                'opportunity', 'deal', 'pipeline', 'stage', 'probability', 'close_date', 'forecast',
                'opp_id', 'deal_id', 'sales_stage', 'win_probability', 'expected_revenue', 'deal_value'
            ],
            'contacts': [
                'contact', 'person', 'individual', 'first_name', 'last_name', 'title', 'relationship',
                'contact_id', 'person_id', 'full_name', 'job_title', 'phone', 'mobile', 'email'
            ],
            'companies': [
                'company', 'organization', 'enterprise', 'business', 'industry', 'sector', 'headquarters',
                'company_id', 'org_id', 'business_name', 'company_name', 'industry_type', 'company_size'
            ],
            'activities': [
                'activity', 'action', 'event', 'log', 'history', 'timeline', 'interaction',
                'activity_id', 'event_id', 'action_type', 'activity_type', 'interaction_type', 'follow_up'
            ],
            'meetings': [
                'meeting', 'appointment', 'schedule', 'calendar', 'attendee', 'agenda', 'duration',
                'meeting_id', 'appointment_id', 'scheduled', 'start_time', 'end_time', 'location'
            ],
            'campaigns': [
                'campaign', 'marketing', 'promotion', 'advertising', 'channel', 'target', 'response',
                'campaign_id', 'promo_id', 'marketing_campaign', 'campaign_name', 'campaign_type'
            ],
            'tickets': [
                'ticket', 'issue', 'support', 'incident', 'priority', 'resolution', 'escalation',
                'ticket_id', 'issue_id', 'support_ticket', 'incident_id', 'case_id', 'help_desk'
            ],
            'users': [
                'user', 'username', 'login', 'profile', 'role', 'permission', 'access',
                'user_id', 'account', 'user_name', 'login_name', 'user_role', 'access_level'
            ],
            
            # Extended business domains
            'communication': [
                'message', 'email', 'mail', 'subject', 'sender', 'recipient', 'date',
                'corps du message', 'mail expéditeur', 'mail destinataire', 'objet', 'visite_mail'
            ],
            'sales_orders': ['order', 'product', 'price', 'quantity', 'total', 'invoice', 'payment'],
            'financial': ['amount', 'cost', 'revenue', 'budget', 'transaction', 'account', 'currency'],
            'hr_employee': ['employee', 'staff', 'salary', 'department', 'position', 'hire', 'manager'],
            'inventory': ['item', 'stock', 'warehouse', 'supplier', 'category', 'sku', 'unit'],
            'project': ['project', 'task', 'milestone', 'deadline', 'status', 'resource', 'team'],
            'logistics': ['shipment', 'delivery', 'tracking', 'carrier', 'destination', 'weight']
        }
        
        # Calculate domain scores with weighted importance
        domain_scores = {}
        for domain, keywords in domain_patterns.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                for col in columns_lower:
                    if keyword in col:
                        # Weight exact matches higher
                        if keyword == col:
                            score += 3
                        # Weight ID fields higher (strong indicators)
                        elif keyword.endswith('_id') and keyword in col:
                            score += 2.5
                        # Regular substring matches
                        else:
                            score += 1
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
        columns = file_info.get('columns', [])
        file_name = file_info.get('file_name', 'data file')
        file_type = file_info.get('file_type', 'file')
        
        domain_hints = self._infer_data_domain(columns)
        
        # Enhanced query templates with multilingual support
        query_templates = [
            f"database table for storing {domain_hints['primary_domain']} data with fields like {', '.join(columns[:6])}",
            f"{file_type} data ingestion into relational database with columns {', '.join(columns[:8])}",
            f"business data management system for {domain_hints['business_area']} information",
            f"data warehouse table for {domain_hints['data_category']} records and analytics",
            f"structured data storage for {file_name} content in enterprise database",
            "email communication tracking and management system",
            "customer correspondence and interaction history",
            "mail message storage and organization",
            "contact information and relationship management",
            "document and file management system",
            "business communication workflow and tracking"
        ]

        
        if user_context:
            context_query = f"{user_context} with data structure: {', '.join(columns[:10])}"
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
            prompt = f"The file is related to {domain}."
            f"Find the best matching database tables and their relationships in the existing database schema for storing this {domain} data."
            
            logging.info(f"Generated user context: {prompt[:100]}...")
            return prompt
            
        except Exception as e:
            logging.error(f"Error in generate_user_context_by_file_type: {e}")
            return "Importing data from the file into the CRM system by mapping fields to the most relevant database table."

    def search_relevant_tables(self, queries, top_k=15):
        """Search for relevant tables using multiple semantic queries with individual processing"""
        all_results = {}
        
        for i, query in enumerate(queries):
            logging.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                # Individual embedding creation
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
                    
                    if table_name not in all_results:
                        all_results[table_name] = {
                            'table_name': table_name,
                            'table_code': point.payload['table_code'],
                            'table_kind': point.payload['table_kind'],
                            'field_count': point.payload['field_count'],
                            'content': point.payload['content'],
                            'metadata': point.payload['metadata'],
                            'scores': [],
                            'queries_matched': []
                        }
                    
                    all_results[table_name]['scores'].append(point.score)
                    all_results[table_name]['queries_matched'].append(i+1)
                
            except Exception as e:
                logging.error(f"Search error for query {i+1}: {e}")
        
        return all_results
    
    def rank_tables_by_relevance(self, search_results):
        """Rank tables by multiple relevance criteria"""
        ranked_tables = []
        
        for table_name, data in search_results.items():
            scores = data['scores']
            avg_score = sum(scores) / len(scores) if scores else 0
            max_score = max(scores) if scores else 0
            query_coverage = len(set(data['queries_matched']))
            
            composite_score = (max_score * 0.4) + (avg_score * 0.4) + (query_coverage * 0.2)
            
            ranked_tables.append({
                'table_name': table_name,
                'table_code': data['table_code'],
                'table_kind': data['table_kind'],
                'field_count': data['field_count'],
                'content': data['content'],
                'avg_score': avg_score,
                'max_score': max_score,
                'query_coverage': query_coverage,
                'composite_score': composite_score,
                'total_matches': len(scores),
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
                'sql_agent_ready': len(ranked_tables) > 0 and ranked_tables[0]['composite_score'] > 0.6
            }
        }
        
        return final_results

    def _calculate_confidence_level(self, best_table):
        """Calculate confidence level for SQL agent"""
        if not best_table:
            return 'None'
        score = best_table['composite_score']
        if score > 0.8:
            return 'High'
        elif score > 0.6:
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
        logging.info(f"SQL Agent Ready: {'Yes' if summary['sql_agent_ready'] else 'No'}")
        
        logging.info("TOP 10 DATABASE TABLES:") 
        for i, table in enumerate(results['top_10_tables'], 1):  
            logging.info(f"{i}. {table['table_name']} ({table['table_kind']})")
            logging.info(f"   Score: {table['composite_score']:.3f} | Fields: {table['field_count']} | Matches: {table['total_matches']}")
                
        if summary['sql_agent_ready']:
            logging.info("Ready for SQL Agent Processing")
        else:
            logging.warning("Requires review before SQL generation")
    
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
                'ready_for_sql': results['ingestion_summary']['sql_agent_ready']
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
    existing_collections = [col.name for col in qdrant_client.get_collections().collections]
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
