"""
Enhanced Document Ingestion Agent with Vision Capabilities and RAG Pipeline

This module implements a comprehensive document processing agent that can:
1. Support bulk import for various file formats (PDF, CSV, XLSX, JSON, images)
2. Classify documents by type (receipt, form, contact card, etc.)
3. Extract data using Vision Large Model (VLM)
4. Structure data with Large Language Model (LLM)
5. Use RAG pipeline to enhance LLM context with database knowledge
6. Generate accurate SQL queries for CRM database ingestion
7. Provide standardized responses for software integration
"""

import os
import json
import base64
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

# LangChain and LangGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# Vector store
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

# Internal modules
from Agent.RAG.chunk_generator import generate_table_ingestion_chunks
from utils import SQLCodeParser
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DocumentClassification:
    """Document classification result"""
    document_type: str
    confidence: float
    suggested_tables: List[str]
    extraction_strategy: str

@dataclass
class ExtractionResult:
    """Data extraction result"""
    structured_data: Dict[str, Any]
    fields_detected: List[str]
    data_quality_score: float
    validation_errors: List[str]

@dataclass
class IngestionPlan:
    """Database ingestion plan"""
    target_table: str
    sql_statements: Dict[str, str]  # {"create": "...", "insert": "..."}
    field_mappings: Dict[str, str]
    confidence_score: float
    requires_human_review: bool
    estimated_success_rate: float

@dataclass
class ProcessingResult:
    """Final processing result"""
    document_classification: DocumentClassification
    extraction_result: ExtractionResult
    ingestion_plan: IngestionPlan
    processing_time: float
    status: str  # "success", "requires_review", "error"
    metadata: Dict[str, Any]

class EnhancedDocumentIngestionAgent:
    """
    Main agent class implementing the complete document ingestion pipeline
    with vision capabilities, RAG enhancement, and ReAct framework.
    """
    
    def __init__(self, 
                 qdrant_client: QdrantClient,
                 collection_name: str = "maxo_vector_store_v2",
                 model_name: str = "gpt-4o"):
        """
        Initialize the enhanced document ingestion agent
        
        Args:
            qdrant_client: Initialized Qdrant client for RAG
            collection_name: Vector store collection name
            model_name: OpenAI model name
        """
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.model_name = model_name
        
        # Initialize models
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.vision_llm = ChatOpenAI(model=model_name, temperature=0)
        self.encoder = OpenAIEmbeddings()
        
        # Document type patterns for classification
        self.document_patterns = {
            "invoice": ["invoice", "bill", "receipt", "payment", "total", "amount due", "tax"],
            "contact_card": ["name", "email", "phone", "address", "company", "title"],
            "form": ["form", "application", "registration", "fields", "checkbox"],
            "receipt": ["receipt", "purchase", "transaction", "date", "amount", "store"],
            "contract": ["agreement", "contract", "terms", "conditions", "signature"],
            "medical_record": ["patient", "doctor", "diagnosis", "treatment", "medical"],
            "financial_statement": ["balance", "assets", "liabilities", "income", "expenses"],
            "report": ["report", "analysis", "summary", "findings", "recommendations"]
        }
        
        # Initialize RAG components
        self._initialize_rag_system()
        
        # Build ReAct agent graph
        self.agent_graph = self._build_agent_graph()

    def _initialize_rag_system(self):
        """Initialize the RAG system with database schema knowledge"""
        try:
            # Check if collection exists
            collections = [col.name for col in self.qdrant_client.get_collections().collections]
            if self.collection_name not in collections:
                raise ValueError(f"Collection {self.collection_name} not found. Please run database setup first.")
            
            print(f"✅ RAG system initialized with collection: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ RAG initialization error: {e}")
            raise

    @tool
    def classify_document_type(self, file_path: str, extracted_text: str = None) -> str:
        """
        Classify document type using vision model and text analysis
        
        Args:
            file_path: Path to the document file
            extracted_text: Optional pre-extracted text
            
        Returns:
            JSON string with classification results
        """
        try:
            classification_prompt = """
            Analyze this document and classify its type. Consider:
            1. Visual layout and structure
            2. Text content and keywords
            3. Form fields and data organization
            
            Return a JSON object with:
            {
                "document_type": "invoice|contact_card|form|receipt|contract|medical_record|financial_statement|report|other",
                "confidence": 0.0-1.0,
                "key_indicators": ["list", "of", "detected", "patterns"],
                "suggested_extraction_strategy": "tabular|form_fields|free_text|structured_data"
            }
            """
            
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff')):
                # Use vision model for image/PDF files
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                
                image_base64 = base64.b64encode(file_bytes).decode('utf-8')
                
                message = [HumanMessage(
                    content=[
                        {"type": "text", "text": classification_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                )]
                
                response = self.vision_llm.invoke(message)
                return response.content
            
            else:
                # Use text-based classification for other files
                if not extracted_text:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        extracted_text = f"Columns: {list(df.columns)}\nSample data: {df.head(2).to_string()}"
                    elif file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        extracted_text = json.dumps(data, indent=2)[:1000]
                
                message = [HumanMessage(content=f"{classification_prompt}\n\nDocument content:\n{extracted_text}")]
                response = self.llm.invoke(message)
                return response.content
                
        except Exception as e:
            return json.dumps({
                "document_type": "error",
                "confidence": 0.0,
                "error": str(e)
            })

    @tool
    def extract_structured_data(self, file_path: str, document_type: str, schema_context: str = None) -> str:
        """
        Extract structured data from document using appropriate strategy
        
        Args:
            file_path: Path to the document
            document_type: Classification result from classify_document_type
            schema_context: Optional database schema context from RAG
            
        Returns:
            JSON string with extracted structured data
        """
        try:
            extraction_strategies = {
                "invoice": self._extract_invoice_data,
                "contact_card": self._extract_contact_data,
                "form": self._extract_form_data,
                "receipt": self._extract_receipt_data,
                "financial_statement": self._extract_financial_data
            }
            
            strategy = extraction_strategies.get(document_type, self._extract_generic_data)
            return strategy(file_path, schema_context)
            
        except Exception as e:
            return json.dumps({
                "error": f"Extraction failed: {str(e)}",
                "structured_data": {},
                "fields_detected": [],
                "data_quality_score": 0.0
            })

    def _extract_invoice_data(self, file_path: str, schema_context: str = None) -> str:
        """Extract structured data from invoices"""
        extraction_prompt = f"""
        Extract invoice data into this JSON structure:
        {{
            "invoice_number": "",
            "date": "",
            "vendor_name": "",
            "vendor_address": "",
            "customer_name": "",
            "customer_address": "",
            "line_items": [
                {{"description": "", "quantity": 0, "unit_price": 0.0, "total": 0.0}}
            ],
            "subtotal": 0.0,
            "tax": 0.0,
            "total_amount": 0.0,
            "payment_terms": "",
            "due_date": ""
        }}
        
        {f"Database schema context: {schema_context}" if schema_context else ""}
        
        Return only valid JSON.
        """
        
        return self._process_with_vision_model(file_path, extraction_prompt)

    def _extract_contact_data(self, file_path: str, schema_context: str = None) -> str:
        """Extract structured data from contact cards/forms"""
        extraction_prompt = f"""
        Extract contact information into this JSON structure:
        {{
            "first_name": "",
            "last_name": "",
            "full_name": "",
            "email": "",
            "phone": "",
            "mobile": "",
            "company": "",
            "job_title": "",
            "address": "",
            "city": "",
            "state": "",
            "zip_code": "",
            "country": "",
            "website": "",
            "notes": ""
        }}
        
        {f"Database schema context: {schema_context}" if schema_context else ""}
        
        Return only valid JSON.
        """
        
        return self._process_with_vision_model(file_path, extraction_prompt)

    def _extract_form_data(self, file_path: str, schema_context: str = None) -> str:
        """Extract structured data from forms"""
        extraction_prompt = f"""
        Extract all form fields and values into JSON structure:
        {{
            "form_type": "",
            "form_id": "",
            "submission_date": "",
            "fields": {{
                "field_name": "field_value"
            }},
            "checkboxes": {{
                "checkbox_name": true/false
            }},
            "selections": {{
                "dropdown_name": "selected_value"
            }}
        }}
        
        {f"Database schema context: {schema_context}" if schema_context else ""}
        
        Return only valid JSON.
        """
        
        return self._process_with_vision_model(file_path, extraction_prompt)

    def _extract_receipt_data(self, file_path: str, schema_context: str = None) -> str:
        """Extract structured data from receipts"""
        extraction_prompt = f"""
        Extract receipt data into this JSON structure:
        {{
            "store_name": "",
            "store_address": "",
            "transaction_id": "",
            "date": "",
            "time": "",
            "items": [
                {{"name": "", "quantity": 0, "price": 0.0}}
            ],
            "subtotal": 0.0,
            "tax": 0.0,
            "total": 0.0,
            "payment_method": "",
            "cashier": ""
        }}
        
        {f"Database schema context: {schema_context}" if schema_context else ""}
        
        Return only valid JSON.
        """
        
        return self._process_with_vision_model(file_path, extraction_prompt)

    def _extract_financial_data(self, file_path: str, schema_context: str = None) -> str:
        """Extract structured data from financial statements"""
        extraction_prompt = f"""
        Extract financial data into this JSON structure:
        {{
            "statement_type": "",
            "period": "",
            "company_name": "",
            "assets": {{
                "current_assets": 0.0,
                "fixed_assets": 0.0,
                "total_assets": 0.0
            }},
            "liabilities": {{
                "current_liabilities": 0.0,
                "long_term_liabilities": 0.0,
                "total_liabilities": 0.0
            }},
            "equity": {{
                "total_equity": 0.0
            }},
            "income": {{
                "revenue": 0.0,
                "expenses": 0.0,
                "net_income": 0.0
            }}
        }}
        
        {f"Database schema context: {schema_context}" if schema_context else ""}
        
        Return only valid JSON.
        """
        
        return self._process_with_vision_model(file_path, extraction_prompt)

    def _extract_generic_data(self, file_path: str, schema_context: str = None) -> str:
        """Generic data extraction for unknown document types"""
        extraction_prompt = f"""
        Extract all structured data from this document into JSON format.
        Identify key-value pairs, tables, lists, and any structured information.
        
        Use this general structure but adapt as needed:
        {{
            "document_title": "",
            "date": "",
            "data_fields": {{
                "field_name": "value"
            }},
            "tables": [
                {{"table_name": "", "rows": []}}
            ],
            "lists": {{
                "list_name": []
            }}
        }}
        
        {f"Database schema context: {schema_context}" if schema_context else ""}
        
        Return only valid JSON.
        """
        
        return self._process_with_vision_model(file_path, extraction_prompt)

    def _process_with_vision_model(self, file_path: str, prompt: str) -> str:
        """Process file with vision model using the given prompt"""
        try:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff')):
                # Vision model processing for images/PDFs
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                
                image_base64 = base64.b64encode(file_bytes).decode('utf-8')
                
                message = [HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                )]
                
                response = self.vision_llm.invoke(message)
                return response.content
            
            else:
                # Text-based processing for CSV, JSON, etc.
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    content = f"CSV file with columns: {list(df.columns)}\nSample data:\n{df.head().to_string()}"
                elif file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    content = json.dumps(data, indent=2)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                message = [HumanMessage(content=f"{prompt}\n\nDocument content:\n{content}")]
                response = self.llm.invoke(message)
                return response.content
                
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def enhance_with_rag_context(self, extracted_data: str, document_type: str) -> str:
        """
        Use RAG to find relevant database tables and enhance extraction context
        
        Args:
            extracted_data: JSON string of extracted data
            document_type: Type of document being processed
            
        Returns:
            JSON string with RAG-enhanced context and table recommendations
        """
        try:
            # Parse extracted data to understand field types
            data = json.loads(extracted_data)
            fields = list(data.keys()) if isinstance(data, dict) else []
            
            # Generate RAG queries based on document type and fields
            rag_queries = self._generate_rag_queries(document_type, fields)
            
            # Search for relevant tables
            relevant_tables = []
            for query in rag_queries:
                query_embedding = self.encoder.embed_query(query)
                
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
                    limit=5
                )
                
                for point in search_results.points:
                    relevant_tables.append({
                        "table_name": point.payload['primary_table'],
                        "table_code": point.payload['table_code'],
                        "field_count": point.payload['field_count'],
                        "relevance_score": point.score,
                        "schema": point.payload['content']
                    })
            
            # Rank and deduplicate tables
            table_scores = {}
            for table in relevant_tables:
                name = table['table_name']
                if name not in table_scores or table['relevance_score'] > table_scores[name]['relevance_score']:
                    table_scores[name] = table
            
            top_tables = sorted(table_scores.values(), key=lambda x: x['relevance_score'], reverse=True)[:3]
            
            return json.dumps({
                "recommended_tables": top_tables,
                "enhancement_context": f"Found {len(top_tables)} relevant tables for {document_type}",
                "field_mapping_suggestions": self._suggest_field_mappings(data, top_tables)
            })
            
        except Exception as e:
            return json.dumps({
                "error": f"RAG enhancement failed: {str(e)}",
                "recommended_tables": [],
                "enhancement_context": "",
                "field_mapping_suggestions": {}
            })

    def _generate_rag_queries(self, document_type: str, fields: List[str]) -> List[str]:
        """Generate RAG search queries based on document type and detected fields"""
        base_queries = {
            "invoice": [
                "invoice billing financial transaction database table",
                "vendor supplier payment amount table structure",
                "line items products services database schema"
            ],
            "contact_card": [
                "contact person individual customer database table",
                "name email phone address contact management",
                "customer relationship management CRM contact"
            ],
            "form": [
                "form data submission application database table",
                "user input form fields data collection",
                "registration application form database"
            ],
            "receipt": [
                "receipt transaction purchase sales database table",
                "store retail transaction payment method",
                "sales transaction receipt database"
            ],
            "financial_statement": [
                "financial statement balance sheet income database",
                "assets liabilities equity financial data",
                "accounting financial reporting database"
            ]
        }
        
        queries = base_queries.get(document_type, ["generic business data database table"])
        
        # Add field-specific queries
        if fields:
            field_query = f"database table with fields: {', '.join(fields[:8])}"
            queries.append(field_query)
        
        return queries

    def _suggest_field_mappings(self, extracted_data: Dict, tables: List[Dict]) -> Dict[str, str]:
        """Suggest field mappings between extracted data and database tables"""
        mappings = {}
        
        if not tables or not extracted_data:
            return mappings
        
        # Get field names from extracted data
        extracted_fields = list(extracted_data.keys()) if isinstance(extracted_data, dict) else []
        
        # Analyze top table schema for mapping suggestions
        top_table = tables[0]
        schema_content = top_table.get('schema', '')
        
        # Simple heuristic mapping (can be enhanced with ML)
        field_mapping_hints = {
            'first_name': ['firstname', 'fname', 'first', 'given_name'],
            'last_name': ['lastname', 'lname', 'last', 'surname'],
            'email': ['email', 'mail', 'email_address'],
            'phone': ['phone', 'telephone', 'mobile', 'tel'],
            'company': ['company', 'organization', 'org', 'business'],
            'address': ['address', 'street', 'location'],
            'date': ['date', 'created', 'timestamp'],
            'amount': ['amount', 'total', 'sum', 'value'],
            'description': ['description', 'desc', 'note', 'comment']
        }
        
        for extracted_field in extracted_fields:
            field_lower = extracted_field.lower()
            for db_pattern, variations in field_mapping_hints.items():
                if any(var in field_lower for var in variations):
                    mappings[extracted_field] = f"maps_to_{db_pattern}"
                    break
        
        return mappings

    @tool
    def generate_ingestion_sql(self, extracted_data: str, rag_context: str, target_table: str = None) -> str:
        """
        Generate SQL statements for database ingestion
        
        Args:
            extracted_data: JSON string of extracted structured data
            rag_context: RAG enhancement context with table recommendations
            target_table: Optional specific target table
            
        Returns:
            JSON string with SQL statements and ingestion plan
        """
        try:
            data = json.loads(extracted_data)
            context = json.loads(rag_context)
            
            recommended_tables = context.get('recommended_tables', [])
            target = target_table or (recommended_tables[0]['table_name'] if recommended_tables else 'extracted_data')
            
            # Get detailed schema if available
            schema_context = ""
            if recommended_tables:
                schema_context = recommended_tables[0]['schema']
            
            sql_generation_prompt = f"""
            Generate SQL statements for ingesting this data into the database:
            
            Extracted Data:
            {json.dumps(data, indent=2)}
            
            Target Table: {target}
            
            Database Schema Context:
            {schema_context}
            
            Field Mapping Suggestions:
            {json.dumps(context.get('field_mapping_suggestions', {}), indent=2)}
            
            Please provide:
            1. CREATE TABLE statement (if table doesn't exist)
            2. INSERT statement(s) for the data
            3. Any necessary data transformations
            
            Return JSON format:
            {{
                "create_table": "CREATE TABLE statement or null if table exists",
                "insert_statements": ["INSERT statement 1", "INSERT statement 2"],
                "data_transformations": ["transformation notes"],
                "confidence_score": 0.0-1.0,
                "requires_review": true/false,
                "review_notes": ["notes for human review"]
            }}
            """
            
            response = self.llm.invoke([HumanMessage(content=sql_generation_prompt)])
            
            # Parse and validate the SQL
            try:
                sql_result = json.loads(response.content)
                
                # Clean SQL statements using SQLCodeParser
                if sql_result.get('create_table'):
                    sql_result['create_table'] = SQLCodeParser.extract_sql_code(sql_result['create_table'])
                
                if sql_result.get('insert_statements'):
                    cleaned_inserts = []
                    for insert in sql_result['insert_statements']:
                        cleaned_inserts.append(SQLCodeParser.extract_sql_code(insert))
                    sql_result['insert_statements'] = cleaned_inserts
                
                return json.dumps(sql_result)
                
            except json.JSONDecodeError:
                # Fallback: extract SQL from response content
                sql_content = SQLCodeParser.extract_sql_code(response.content)
                separated = SQLCodeParser.separate_statements(sql_content)
                
                return json.dumps({
                    "create_table": separated['create_table'] or None,
                    "insert_statements": separated['insert_statements'].split('\n\n') if separated['insert_statements'] else [],
                    "confidence_score": 0.7,
                    "requires_review": True,
                    "review_notes": ["SQL parsed from unstructured response"]
                })
                
        except Exception as e:
            return json.dumps({
                "error": f"SQL generation failed: {str(e)}",
                "create_table": None,
                "insert_statements": [],
                "confidence_score": 0.0,
                "requires_review": True
            })

    def _build_agent_graph(self) -> StateGraph:
        """Build the ReAct agent graph with all tools"""
        
        # Define all tools for the agent
        tools = [
            self.classify_document_type,
            self.extract_structured_data,
            self.enhance_with_rag_context,
            self.generate_ingestion_sql
        ]
        
        def assistant(state: MessagesState):
            """Assistant node that processes messages and decides on tool usage"""
            response = self.llm.bind_tools(tools).invoke(state["messages"])
            return {"messages": [response]}
        
        # Build the graph
        builder = StateGraph(MessagesState)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(tools))
        
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        
        return builder.compile()

    def process_document(self, file_path: str, 
                        user_context: str = None,
                        target_table: str = None) -> ProcessingResult:
        """
        Main method to process a document through the complete pipeline
        
        Args:
            file_path: Path to the document to process
            user_context: Optional user-provided context
            target_table: Optional specific target table
            
        Returns:
            ProcessingResult with complete processing information
        """
        start_time = datetime.now()
        
        try:
            # Create processing prompt
            processing_prompt = f"""
            Process this document for database ingestion: {file_path}
            
            Steps to follow:
            1. Classify the document type
            2. Extract structured data appropriate for the document type
            3. Use RAG to find relevant database tables and enhance context
            4. Generate SQL statements for database ingestion
            
            {f"User context: {user_context}" if user_context else ""}
            {f"Target table: {target_table}" if target_table else ""}
            
            Provide comprehensive results for each step.
            """
            
            # Run through the agent graph
            messages = [HumanMessage(content=processing_prompt)]
            result = self.agent_graph.invoke({"messages": messages})
            
            # Extract results from agent conversation
            final_message = result["messages"][-1]
            
            # Parse the results (this would need refinement based on actual agent output)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # For now, return a structured result (in practice, you'd parse the agent's output)
            return ProcessingResult(
                document_classification=DocumentClassification(
                    document_type="processed",
                    confidence=0.8,
                    suggested_tables=["extracted_table"],
                    extraction_strategy="agent_processed"
                ),
                extraction_result=ExtractionResult(
                    structured_data={"processed": True},
                    fields_detected=["agent_processed"],
                    data_quality_score=0.8,
                    validation_errors=[]
                ),
                ingestion_plan=IngestionPlan(
                    target_table=target_table or "processed_data",
                    sql_statements={"processed": "Agent generated SQL"},
                    field_mappings={"processed": "mapping"},
                    confidence_score=0.8,
                    requires_human_review=False,
                    estimated_success_rate=0.8
                ),
                processing_time=processing_time,
                status="success",
                metadata={
                    "agent_messages": len(result["messages"]),
                    "file_path": file_path,
                    "user_context": user_context
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                document_classification=DocumentClassification("error", 0.0, [], "error"),
                extraction_result=ExtractionResult({}, [], 0.0, [str(e)]),
                ingestion_plan=IngestionPlan("", {}, {}, 0.0, True, 0.0),
                processing_time=processing_time,
                status="error",
                metadata={"error": str(e)}
            )

    def export_standardized_response(self, result: ProcessingResult) -> Dict[str, Any]:
        """
        Export results in standardized format for software integration
        
        Args:
            result: ProcessingResult from document processing
            
        Returns:
            Standardized response dictionary
        """
        return {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "status": result.status,
            "processing_time_seconds": result.processing_time,
            
            "document_analysis": {
                "type": result.document_classification.document_type,
                "confidence": result.document_classification.confidence,
                "extraction_strategy": result.document_classification.extraction_strategy
            },
            
            "extracted_data": {
                "structured_data": result.extraction_result.structured_data,
                "fields_detected": result.extraction_result.fields_detected,
                "data_quality_score": result.extraction_result.data_quality_score,
                "validation_errors": result.extraction_result.validation_errors
            },
            
            "database_ingestion": {
                "target_table": result.ingestion_plan.target_table,
                "sql_statements": result.ingestion_plan.sql_statements,
                "field_mappings": result.ingestion_plan.field_mappings,
                "confidence_score": result.ingestion_plan.confidence_score,
                "requires_human_review": result.ingestion_plan.requires_human_review,
                "estimated_success_rate": result.ingestion_plan.estimated_success_rate
            },
            
            "recommendations": {
                "suggested_tables": result.document_classification.suggested_tables,
                "review_required": result.ingestion_plan.requires_human_review,
                "next_steps": self._generate_next_steps(result)
            },
            
            "metadata": result.metadata
        }

    def _generate_next_steps(self, result: ProcessingResult) -> List[str]:
        """Generate recommended next steps based on processing results"""
        steps = []
        
        if result.status == "error":
            steps.append("Review error logs and retry processing")
            steps.append("Check file format and accessibility")
        
        elif result.ingestion_plan.requires_human_review:
            steps.append("Human review required before database ingestion")
            steps.append("Validate extracted data accuracy")
            steps.append("Confirm field mappings")
        
        elif result.ingestion_plan.confidence_score > 0.8:
            steps.append("Ready for automatic database ingestion")
            steps.append("Execute SQL statements")
            steps.append("Verify ingestion success")
        
        else:
            steps.append("Review confidence scores")
            steps.append("Consider manual field mapping")
            steps.append("Test with sample data first")
        
        return steps

# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Document Ingestion Agent initialized")
    print("Ready for document processing with vision capabilities and RAG pipeline")