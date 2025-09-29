import os
import pandas as pd
import json
import base64
from typing import Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

# Fix relative imports to work both as module and standalone
try:
    from .utils import SQLCodeParser
except ImportError:
    try:
        from utils import SQLCodeParser
    except ImportError:
        # Create a dummy SQLCodeParser if utils is not available
        class SQLCodeParser:
            @staticmethod
            def parse(code):
                return {"queries": [], "errors": []}

# Import RAG components with better error handling
try:
    # Try relative imports first
    try:
        from .RAG.RAG_maxo_database import GenericFileIngestionRAGPipeline
        from .RAG.config import config
    except ImportError:
        # Fall back to absolute imports
        from RAG.RAG_maxo_database import GenericFileIngestionRAGPipeline
        from RAG.config import config
        
    from qdrant_client import QdrantClient
    from langchain_openai import OpenAIEmbeddings

    load_dotenv(r"C:\Users\axel.grille\Documents\rules-engine-agent\Agent\.env")
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    rag_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    rag_qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Use the same collection name as in your RAG script
    rag_collection_name = "maxo_vector_store_v2"
    
    RAG_AVAILABLE = True
except Exception as e:
    print(f"Warning: RAG components not available: {e}")
    RAG_AVAILABLE = False
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or load_dotenv() and os.getenv("OPENAI_API_KEY")

# PDF processing imports
try:
    import PyPDF2
    import pdfplumber
    import fitz  
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

vision_llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

@tool
def analyze_file(file_path: str) -> str:
    """
    Analyze various file types including PDF, CSV, Excel, JSON, images, and text files.
    Extracts content and provides structured analysis for database ingestion.
    
    Args:
        file_path: Path to the file to analyze
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if (file_extension == '.pdf'):
            return _analyze_pdf(file_path)
        elif file_extension in ['.csv']:
            return _analyze_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return _analyze_excel(file_path)
        elif file_extension == '.json':
            return _analyze_json(file_path)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            return _analyze_image(file_path)
        elif file_extension in ['.txt', '.xml']:
            return _analyze_text(file_path)
        else:
            return f"Unsupported file type: {file_extension}"
    
    except Exception as e:
        return f"Error analyzing file: {str(e)}"

def _analyze_pdf(file_path: str) -> str:
    """Analyze PDF files with text extraction and OCR fallback."""
    if not PDF_AVAILABLE:
        return "Error: PDF processing libraries not installed. Please install PyPDF2, pdfplumber, and PyMuPDF."
    
    results = []
    text_content = ""
    
    # Try text extraction first
    try:
        with pdfplumber.open(file_path) as pdf:
            text_pages = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(f"Page {i+1}:\n{page_text}")
            
            if text_pages:
                text_content = "\n\n".join(text_pages)
    except Exception as e:
        results.append(f"Text extraction error: {str(e)}")
    
    # No text, try OCR
    if not text_content.strip():
        try:
            doc = fitz.open(file_path)
            ocr_pages = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                
                # Use vision model for better OCR
                image_base64 = base64.b64encode(img_data).decode('utf-8')
                ocr_text = _extract_text_from_image_base64(image_base64)
                if ocr_text:
                    ocr_pages.append(f"Page {page_num+1} (OCR):\n{ocr_text}")
            
            if ocr_pages:
                text_content = "\n\n".join(ocr_pages)
            doc.close()
        except Exception as e:
            results.append(f"OCR error: {str(e)}")
    
    if text_content:
        results.append(f"PDF Content Analysis:\n{text_content[:2000]}...")
        
        # Analyze structure for database design
        lines = text_content.split('\n')
        potential_headers = [line.strip() for line in lines[:20] if line.strip() and len(line.strip()) < 100]
        
        results.append(f"\nStructural Analysis:")
        results.append(f"- Total pages: {len(text_content.split('Page '))}")
        results.append(f"- Potential headers/fields: {potential_headers[:5]}")
        
        # Look for tabular data patterns
        table_patterns = []
        for line in lines:
            if '|' in line or '\t' in line or ',' in line:
                table_patterns.append(line.strip())
        
        if table_patterns:
            results.append(f"- Potential tabular data found: {len(table_patterns)} lines")
            results.append(f"- Sample: {table_patterns[0][:100]}...")
    
    else:
        results.append("No text content could be extracted from the PDF")
    
    return "\n".join(results)

def _analyze_csv(file_path: str) -> str:
    """Analyze CSV files for database mapping discovery."""
    try:
        # Add delimiter detection for semicolon-separated files
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        
        delimiter = '\t' if '\t' in first_line else ',' if ',' in first_line else ';'
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        results = []
        results.append(f"CSV Analysis for: {os.path.basename(file_path)}")
        results.append(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
        results.append(f"Columns: {list(df.columns)}")
        
        # Data types analysis for mapping purposes
        results.append("\nData Types Analysis:")
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            # Infer general data type category for mapping
            if dtype == 'object':
                max_length = df[col].astype(str).str.len().max()
                data_type = f"Text (max length: {max_length})"
            elif dtype in ['int64', 'int32']:
                data_type = "Integer"
            elif dtype in ['float64', 'float32']:
                data_type = "Decimal/Float"
            elif 'datetime' in str(dtype):
                data_type = "DateTime"
            else:
                data_type = "Mixed/Other"
            
            results.append(f"  {col}: {data_type} (nulls: {null_count}, unique: {unique_count})")
        
        # Sample data
        results.append(f"\nSample Data (first 3 rows):")
        results.append(df.head(3).to_string())
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"

def _analyze_excel(file_path: str) -> str:
    """Analyze Excel files for database mapping discovery."""
    try:
        # Get all sheet names
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        results = []
        results.append(f"Excel Analysis for: {os.path.basename(file_path)}")
        results.append(f"Sheets found: {sheet_names}")
        
        # Analyze each sheet
        for sheet_name in sheet_names[:3]:  # Limit to first 3 sheets
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            results.append(f"\n--- Sheet: {sheet_name} ---")
            results.append(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
            results.append(f"Columns: {list(df.columns)}")
            
            # Data types for first sheet only (to avoid too much output)
            if sheet_name == sheet_names[0]:
                results.append("\nData Types Analysis:")
                for col in df.columns:
                    dtype = df[col].dtype
                    null_count = df[col].isnull().sum()
                    unique_count = df[col].nunique()
                    
                    # Infer general data type category for mapping
                    if dtype == 'object':
                        max_length = df[col].astype(str).str.len().max()
                        data_type = f"Text (max length: {max_length})"
                    elif dtype in ['int64', 'int32']:
                        data_type = "Integer"
                    elif dtype in ['float64', 'float32']:
                        data_type = "Decimal/Float"
                    elif 'datetime' in str(dtype):
                        data_type = "DateTime"
                    else:
                        data_type = "Mixed/Other"
                    
                    results.append(f"  {col}: {data_type} (nulls: {null_count}, unique: {unique_count})")
                
                results.append(f"\nSample Data:")
                results.append(df.head(2).to_string())
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error analyzing Excel: {str(e)}"

def _analyze_json(file_path: str) -> str:
    """Analyze JSON files for database schema generation."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        results.append(f"JSON Analysis for: {os.path.basename(file_path)}")
        
        if isinstance(data, list):
            results.append(f"Type: Array with {len(data)} items")
            if data:
                sample_item = data[0]
                results.append(f"Sample item structure: {list(sample_item.keys()) if isinstance(sample_item, dict) else type(sample_item)}")
        elif isinstance(data, dict):
            results.append(f"Type: Object with {len(data)} keys")
            results.append(f"Keys: {list(data.keys())}")
        
        # Show structure
        results.append(f"\nStructure Preview:")
        results.append(json.dumps(data, indent=2)[:1000] + "..." if len(str(data)) > 1000 else json.dumps(data, indent=2))
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error analyzing JSON: {str(e)}"

def _analyze_image(file_path: str) -> str:
    """Analyze images using vision model for structured data extraction."""
    try:
        with open(file_path, 'rb') as f:
            image_bytes = f.read()
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image for structured data extraction and database ingestion. "
                            "If this is a document (invoice, form, receipt, etc.), extract all text and organize it into structured fields. "
                            "If it contains tabular data, identify columns and rows. "
                            "Provide the extracted information in a clear, structured format suitable for database storage. "
                            "Also suggest appropriate database field names and data types."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        }
                    }
                ]
            )
        ]
        
        response = vision_llm.invoke(message)
        return f"Image Analysis for: {os.path.basename(file_path)}\n\n{response.content}"
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def _extract_text_from_image_base64(image_base64: str) -> str:
    """Extract text from image using vision model."""
    try:
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Extract all text from this image. Return only the text content, no explanations.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        }
                    }
                ]
            )
        ]
        
        response = vision_llm.invoke(message)
        return response.content
    except:
        return ""

def _analyze_text(file_path: str) -> str:
    """Analyze text and XML files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = []
        results.append(f"Text Analysis for: {os.path.basename(file_path)}")
        results.append(f"File size: {len(content)} characters")
        results.append(f"Lines: {len(content.split(chr(10)))}")
        
        # Show preview
        results.append(f"\nContent Preview:")
        results.append(content[:1000] + "..." if len(content) > 1000 else content)
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error analyzing text file: {str(e)}"

@tool
def python_code_executor(code: str) -> str:
    """
    Execute Python code for data processing and analysis.
    
    Args:
        code: Python code to execute
    """
    try:
        # Create a safe execution environment
        exec_globals = {
            'pd': pd,
            'json': json,
            'os': os,
            '__builtins__': __builtins__
        }
        exec_locals = {}
        
        exec(code, exec_globals, exec_locals)
        
        # Return any printed output or results
        return "Code executed successfully. Check variables in exec_locals if needed."
    
    except Exception as e:
        return f"Error executing code: {str(e)}"

@tool
def find_matching_database_tables(file_path: str, user_context: str = None) -> str:
    """
    Find the most relevant database tables for ingesting data from a file using RAG.
    Analyzes file structure and matches it against existing database schema using semantic search.
    
    Args:
        file_path: Path to the file to analyze for database ingestion
        user_context: Optional context about the data or intended use
    
    Returns:
        JSON string with top 10 matching tables, confidence scores, and ingestion recommendations
    """
    if not RAG_AVAILABLE:
        # MOCK MODE: Return sample results for testing when RAG is not available
        print("⚠️  RAG not available - using mock data for demonstration")
        
        # Analyze the file to get basic info
        file_analysis_result = analyze_file.invoke({"file_path": file_path})
        filename = os.path.basename(file_path)
        
        # Extract basic file info
        lines = file_analysis_result.split('\n')
        columns = []
        total_rows = 0
        total_columns = 0
        
        for line in lines:
            if line.startswith("Columns: "):
                # Extract columns from the analysis
                columns_str = line.replace("Columns: ", "").strip()
                if columns_str.startswith('[') and columns_str.endswith(']'):
                    columns = eval(columns_str)  # Safe for known format
            elif line.startswith("Dimensions: "):
                # Extract dimensions
                parts = line.split()
                if len(parts) >= 4:
                    total_rows = int(parts[1])
                    total_columns = int(parts[3])
        
        # Generate mock matching results based on filename patterns
        mock_tables = []
        
        if "oppo" in filename.lower():
            mock_tables = [
                {"rank": 1, "table_name": "Opportunity", "table_code": "oppo", "table_kind": "Entity", "composite_score": 0.92, "field_count": 25},
                {"rank": 2, "table_name": "Lead", "table_code": "lead", "table_kind": "Entity", "composite_score": 0.85, "field_count": 18},
                {"rank": 3, "table_name": "Deal", "table_code": "deal", "table_kind": "Entity", "composite_score": 0.78, "field_count": 22}
            ]
        elif "combi" in filename.lower():
            mock_tables = [
                {"rank": 1, "table_name": "Contact", "table_code": "cont", "table_kind": "Entity", "composite_score": 0.88, "field_count": 30},
                {"rank": 2, "table_name": "Company", "table_code": "comp", "table_kind": "Entity", "composite_score": 0.82, "field_count": 25},
                {"rank": 3, "table_name": "Account", "table_code": "acct", "table_kind": "Entity", "composite_score": 0.75, "field_count": 20}
            ]
        else:
            mock_tables = [
                {"rank": 1, "table_name": "GenericData", "table_code": "data", "table_kind": "Entity", "composite_score": 0.70, "field_count": 15},
                {"rank": 2, "table_name": "ImportData", "table_code": "import", "table_kind": "Entity", "composite_score": 0.65, "field_count": 12}
            ]
        
        # Add schema previews
        for table in mock_tables:
            table["schema_preview"] = f"Mock schema for {table['table_name']} table with {table['field_count']} fields..."
        
        summary = {
            "file_analysis": {
                "file_name": filename,
                "file_type": "CSV",
                "total_rows": total_rows,
                "total_columns": total_columns,
                "columns": columns
            },
            "domain_detected": "business data" if "oppo" in filename.lower() or "combi" in filename.lower() else "general data",
            "recommended_table": mock_tables[0]["table_name"] if mock_tables else "GenericData",
            "confidence_level": "High",
            "mapping_ready": True,
            "requires_review": False,
            "top_10_tables": mock_tables
        }
        
        return json.dumps(summary, indent=2)
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    try:
        # Check if collection exists
        existing_collections = [col.name for col in rag_qdrant_client.get_collections().collections]
        if rag_collection_name not in existing_collections:
            return f"Error: Database schema collection '{rag_collection_name}' not found. Please run the RAG feed mode first to populate the vector store."
        
        # Initialize pipeline with query-only mode
        pipeline = GenericFileIngestionRAGPipeline(
            rag_qdrant_client, 
            rag_embeddings, 
            rag_collection_name,
            query_only=True  # Force query-only mode
        )
        
        # Generate user context if not provided
        if not user_context:
            user_context = pipeline.generate_user_context_by_file_type(file_path)
        
        # Run Entity-First pipeline
        results = pipeline.run_entity_first_pipeline(file_path, user_context)
        
        if 'error' in results:
            return f"Error: {results['error']}"
        
        # Add debugging to see what's actually in the results
        try:
            # Extract entities and relations from Entity-First pipeline results
            memory_summary = results.get('memory_summary', {})
            all_entities = memory_summary.get('all_entities', [])
            all_relations = memory_summary.get('all_relations', [])
            
            # Combine entities and relations into top_10_tables format for backward compatibility
            all_tables = []
            
            # Add entities first (higher priority)
            for entity in all_entities:
                all_tables.append({
                    'table_name': entity.get('table_name', ''),
                    'table_code': entity.get('table_code', ''),
                    'table_kind': entity.get('table_kind', ''),
                    'field_count': entity.get('field_count', 0),
                    'content': entity.get('content', ''),
                    'composite_score': entity.get('confidence_score', 0.0)
                })
            
            # Add relations
            for relation in all_relations:
                all_tables.append({
                    'table_name': relation.get('table_name', ''),
                    'table_code': relation.get('table_code', ''),
                    'table_kind': relation.get('table_kind', ''),
                    'field_count': relation.get('field_count', 0),
                    'content': relation.get('content', ''),
                    'composite_score': relation.get('confidence_score', 0.0)
                })
            
            # Sort by composite_score and take top 10
            all_tables.sort(key=lambda x: x['composite_score'], reverse=True)
            top_10_tables = all_tables[:10]
            
            # Format results for agent consumption
            summary = {
                "file_analysis": {
                    "file_name": results.get('file_analysis', {}).get('file_name', ''),
                    "file_type": results.get('file_analysis', {}).get('file_type', ''),
                    "total_rows": results.get('file_analysis', {}).get('total_rows', 0),
                    "total_columns": results.get('file_analysis', {}).get('total_columns', 0),
                    "columns": results.get('file_analysis', {}).get('columns', [])
                },
                "domain_detected": results.get('inferred_domain', {}).get('primary_domain', ''),
                "recommended_table": results.get('ingestion_summary', {}).get('recommended_table', ''),
                "confidence_level": results.get('ingestion_summary', {}).get('confidence_level', ''),
                "mapping_ready": results.get('ingestion_summary', {}).get('mapping_ready', False),
                "requires_review": results.get('ingestion_summary', {}).get('requires_review', True),
                "top_10_tables": [
                    {
                        "rank": i + 1,
                        "table_name": table['table_name'],
                        "table_code": table['table_code'],
                        "table_kind": table['table_kind'],
                        "composite_score": round(table['composite_score'], 3),
                        "field_count": table['field_count'],
                        "schema_preview": table['content'][:500] + "..." if len(table['content']) > 500 else table['content']
                    }
                    for i, table in enumerate(top_10_tables)
                ]
            }
            
            # Export detailed results for further processing
            try:
                export_result = pipeline.export_for_sql_agent(results)
                if 'success' in export_result:
                    summary["detailed_export_path"] = export_result['output_file']
            except:
                pass  # Don't fail if export doesn't work
            
            return json.dumps(summary, indent=2)
            
        except KeyError as ke:
            return f"Error accessing Entity-First results: Missing key '{ke}'"
        except Exception as inner_e:
            return f"Error processing Entity-First results: {str(inner_e)}"
        
    except Exception as e:
        return f"Error running RAG pipeline: {str(e)}"

@tool
def entity_first_database_discovery(file_path: str, user_context: str = None) -> str:
    """
    ENTITY-FIRST DATABASE DISCOVERY: Two-stage pipeline for intelligent table discovery.
    
    Stage 1: Entity-First Search (Higher confidence threshold: 0.7)
    - Searches only Entity tables, excluding Relations  
    - Detects relationship/mapping data automatically
    - Stores high-confidence entities in memory
    
    Stage 2: Relationship Discovery
    - If relationship data detected: Searches Relation tables
    - Otherwise: Finds related tables for best Entity
    - Stores relations with standard threshold (0.6)
    
    Memory Storage: Entities and Relations stored separately for field mapping agent
    
    Args:
        file_path: Path to the file to analyze for database ingestion
        user_context: Optional context about the data or intended use
    
    Returns:
        JSON with discovered entities, relations, relationship flags, and memory summary
    """
    if not RAG_AVAILABLE:
        return "Error: RAG components not available. Please check your environment configuration."
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    try:
        # Check if collection exists
        existing_collections = [col.name for col in rag_qdrant_client.get_collections().collections]
        if rag_collection_name not in existing_collections:
            return f"Error: Database schema collection '{rag_collection_name}' not found. Please run the RAG feed mode first to populate the vector store."
        
        # Initialize pipeline with query-only mode
        pipeline = GenericFileIngestionRAGPipeline(
            rag_qdrant_client, 
            rag_embeddings, 
            rag_collection_name,
            query_only=True
        )
        
        # Generate user context if not provided
        if not user_context:
            user_context = pipeline.generate_user_context_by_file_type(file_path)
        
        # Run Entity-First pipeline
        results = pipeline.run_entity_first_pipeline(file_path, user_context)
        
        if 'error' in results:
            return f"Error: {results['error']}"
        
        # Extract memory summary for response
        memory_summary = results['memory_summary']
        
        # Format results for agent consumption with Entity-First structure
        summary = {
            "pipeline_type": "entity_first_two_stage",
            "file_analysis": {
                "file_name": results['file_analysis']['file_name'],
                "file_type": results['file_analysis']['file_type'],
                "total_rows": results['file_analysis']['total_rows'],
                "total_columns": results['file_analysis']['total_columns'],
                "columns": results['file_analysis']['columns']
            },
            "domain_detected": results['inferred_domain']['primary_domain'],
            "relationship_data_detected": results['relationship_data_detected'],
            
            # Stage 1 Results: Entities
            "entities_discovered": {
                "count": results['entities_discovered'],
                "confidence_threshold": memory_summary['entity_confidence_threshold'],
                "best_entity": memory_summary['best_entity'],
                "all_entities": memory_summary['all_entities'][:5]  # Top 5 entities
            },
            
            # Stage 2 Results: Relations  
            "relations_discovered": {
                "count": results['relations_discovered'],
                "confidence_threshold": memory_summary['relation_confidence_threshold'],
                "all_relations": memory_summary['all_relations'][:5]  # Top 5 relations
            },
            
            # Pipeline Summary
            "ingestion_summary": results['ingestion_summary'],
            
            # Memory Status (for field mapping agent)
            "memory_status": {
                "entities_stored": len(memory_summary['all_entities']),
                "relations_stored": len(memory_summary['all_relations']),
                "ready_for_field_mapping": memory_summary['best_entity'] is not None or len(memory_summary['all_relations']) > 0
            },
            
            # Raw stage results for debugging
            "stage1_entity_results": results.get('stage1_entity_results', []),
            "stage2_relation_results": results.get('stage2_relation_results', [])
        }
        
        # Export detailed results
        export_result = pipeline.export_for_sql_agent(results)
        if 'success' in export_result:
            summary["detailed_export_path"] = export_result['output_file']
        
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        return f"Error running Entity-First RAG pipeline: {str(e)}"

@tool
def intelligent_table_selector(file_analysis: str, rag_results: str, user_preferences: str = None) -> str:
    """
    STEP 3: Intelligently select the best database table and create field mappings.
    Uses advanced reasoning to choose the optimal table and map fields between source and target.
    
    Args:
        file_analysis: Result from analyze_file (Step 1)
        rag_results: Result from find_matching_database_tables (Step 2) 
        user_preferences: Optional user preferences for table selection criteria
    
    Returns:
        JSON with selected table, field mappings, and confidence assessment
    """
    try:
        rag_data = json.loads(rag_results)
        
        prompt = f"""You are an expert database architect tasked with intelligently selecting the best database table for data ingestion and creating precise field mappings.

ORIGINAL FILE ANALYSIS:
{file_analysis}

RAG MATCHING RESULTS:
{json.dumps(rag_data, indent=2)}

USER PREFERENCES: {user_preferences or "None specified"}

TASK: Analyze all available options and make the best decision for:
1. **Table Selection**: Choose the most suitable table from the top_10_tables
2. **Field Mapping**: Create precise mappings between source fields and target table fields
3. **Confidence Assessment**: Evaluate the quality of the match

SELECTION CRITERIA (in priority order):
1. **Schema Compatibility**: Field types and structure alignment
2. **Semantic Match**: Business context and domain alignment  
3. **Data Volume**: Can handle the source data volume
4. **Field Coverage**: Maximum field mapping potential
5. **Confidence Score**: RAG matching confidence

RETURN ONLY A VALID JSON with this exact structure:
{{
    "selected_table": {{
        "table_name": "chosen_table_name",
        "table_code": "table_code", 
        "selection_reason": "detailed explanation of why this table was chosen",
        "confidence_score": 0.85
    }},
    "field_mappings": [
        {{
            "source_field": "original_column_name",
            "target_field": "database_column_name", 
            "data_type": "VARCHAR(100)",
            "transformation": "none|trim|uppercase|date_format|etc",
            "confidence": 0.90
        }}
    ],
    "unmapped_fields": [
        {{
            "source_field": "field_name",
            "reason": "no suitable target found"
        }}
    ],
    "data_quality_concerns": [
        "concern description if any"
    ],
    "ingestion_strategy": "direct|requires_transformation|needs_validation",
    "estimated_success_rate": 0.88
}}"""

        response = vision_llm.invoke([HumanMessage(content=prompt)])
        
        # Extract and validate JSON
        try:
            # Try to extract JSON from response
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            if (json_start != -1 and json_end != 0):
                json_content = response.content[json_start:json_end]
                result = json.loads(json_content)
                return json.dumps(result, indent=2)
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError):
            # Fallback: return structured text result
            return f"Selection Analysis:\n{response.content}"
            
    except Exception as e:
        return f"Error in intelligent table selection: {str(e)}"

@tool
def generate_mapping_visualization(rag_results: str, table_selection: str, file_analysis: str = None) -> str:
    """
    Generate comprehensive visualization of file-to-database structure mapping.
    
    Args:
        rag_results: JSON string from find_matching_database_tables (Step 2)
        table_selection: JSON string from intelligent_table_selector (Step 3)
        file_analysis: Optional detailed file analysis from analyze_file (Step 1)
    
    Returns:
        JSON string with comprehensive mapping visualization between source and target structures
    """
    try:
        # Parse input parameters
        try:
            rag_data = json.loads(rag_results)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid RAG results JSON format"}, indent=2)
        
        # Parse table selection if it's a string
        if isinstance(table_selection, str):
            try:
                table_selection = json.loads(table_selection)
            except json.JSONDecodeError:
                table_selection = {"error": "Could not parse table selection"}
        
        visualization = {
            "source_structure": {},
            "target_structure": {},
            "mapping_details": {},
            "relationship_info": {},
            "summary": {},
            "enhanced_file_details": {}
        }
        
        # Extract source file information from RAG results
        file_info = rag_data.get("file_analysis", {})
        visualization["source_structure"] = {
            "file_name": file_info.get("file_name", "Unknown"),
            "file_type": file_info.get("file_type", "Unknown"),
            "total_rows": file_info.get("total_rows", 0),
            "total_columns": file_info.get("total_columns", 0),
            "columns": file_info.get("columns", []),
            "domain_detected": rag_data.get("domain_detected", "Unknown")
        }
        
        # ENHANCED: Use file_analysis parameter for richer source details
        if file_analysis:
            enhanced_details = _extract_enhanced_file_details(file_analysis)
            visualization["enhanced_file_details"] = enhanced_details
            
            # Merge enhanced details into source structure
            if enhanced_details.get("data_types"):
                visualization["source_structure"]["data_types_analysis"] = enhanced_details["data_types"]
            if enhanced_details.get("sample_data"):
                visualization["source_structure"]["sample_data"] = enhanced_details["sample_data"]
        
        # Extract selected table information
        selected_table_info = table_selection.get("selected_table", {})
        visualization["target_structure"] = {
            "table_name": selected_table_info.get("table_name", "Unknown"),
            "table_code": selected_table_info.get("table_code", "Unknown"),
            "selection_reason": selected_table_info.get("selection_reason", "Not specified"),
            "confidence_score": selected_table_info.get("confidence_score", 0.0)
        }
        
        # Field mappings details
        field_mappings = table_selection.get("field_mappings", [])
        unmapped_fields = table_selection.get("unmapped_fields", [])
        
        visualization["mapping_details"] = {
            "successful_mappings": len(field_mappings),
            "unmapped_fields_count": len(unmapped_fields),
            "field_mappings": field_mappings,
            "unmapped_fields": unmapped_fields,
            "data_quality_concerns": table_selection.get("data_quality_concerns", [])
        }
        
        # Relationship information from RAG results
        top_tables = rag_data.get("top_10_tables", [])
        selected_table_name = selected_table_info.get("table_name")
        
        # Find the selected table in top tables for additional context
        selected_table_details = None
        for table in top_tables:
            if table.get("table_name") == selected_table_name:
                selected_table_details = table
                break
        
        visualization["relationship_info"] = {
            "alternative_tables": [
                {
                    "rank": table.get("rank", 0),
                    "name": table.get("table_name", ""),
                    "score": table.get("composite_score", 0.0),
                    "type": table.get("table_kind", "")
                }
                for table in top_tables[:5]  # Show top 5 alternatives
            ],
            "selected_table_details": selected_table_details
        }
        
        # Summary statistics
        mapping_success_rate = len(field_mappings) / max(len(file_info.get("columns", [])), 1) * 100
        visualization["summary"] = {
            "mapping_success_rate": round(mapping_success_rate, 2),
            "ingestion_strategy": table_selection.get("ingestion_strategy", "unknown"),
            "estimated_success_rate": table_selection.get("estimated_success_rate", 0.0),
            "requires_transformation": any(m.get("transformation", "none") != "none" for m in field_mappings),
            "ready_for_ingestion": mapping_success_rate > 70 and len(table_selection.get("data_quality_concerns", [])) == 0
        }
        
        return json.dumps(visualization, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error generating visualization: {str(e)}"}, indent=2)

def _extract_enhanced_file_details(file_analysis: str) -> dict:
    """Extract enhanced details from file analysis text"""
    enhanced = {}
    
    try:
        lines = file_analysis.split('\n')
        
        # Extract data types if present
        data_types = {}
        in_data_types_section = False
        
        for line in lines:
            line = line.strip()
            if "Data Types Analysis:" in line:
                in_data_types_section = True
                continue
            elif in_data_types_section:
                if line.startswith('  ') and ':' in line:
                    # Parse lines like "  email: VARCHAR(150) (nulls: 0, unique: 1000)"
                    parts = line.split(':')
                    if len(parts) >= 2:
                        field_name = parts[0].strip()
                        type_info = parts[1].strip()
                        data_types[field_name] = type_info
                elif line and not line.startswith('  '):
                    in_data_types_section = False
        
        if data_types:
            enhanced["data_types"] = data_types
        
        # Extract sample data if present
        sample_start = -1
        for i, line in enumerate(lines):
            if "Sample Data" in line and "first" in line:
                sample_start = i + 1
                break
        
        if sample_start > 0 and sample_start < len(lines):
            sample_lines = []
            for i in range(sample_start, min(sample_start + 10, len(lines))):
                if lines[i].strip():
                    sample_lines.append(lines[i])
                else:
                    break
            if sample_lines:
                enhanced["sample_data"] = sample_lines
        
    except Exception as e:
        enhanced["extraction_error"] = str(e)
    
    return enhanced

@tool  
def database_ingestion_orchestrator(file_path: str, user_context: Optional[str] = None, table_name_preference: Optional[str] = None) -> str:
    """
    MAIN ORCHESTRATOR: Execute the complete 4-step database ingestion workflow with validation and refinement.
    
    WORKFLOW:
    1. Analyze file structure and content
    2. Find matching database tables using RAG  
    3. Select optimal table and create detailed field mappings
    4. Validate mapping quality and refine if needed
    
    Args:
        file_path: Path to the file to process
        user_context: Optional context about the data or intended use
        table_name_preference: Optional preference for table naming
    
    Returns:
        Complete workflow results with validation analysis and refinement history
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    workflow_results = {
        "workflow_status": "starting",
        "file_path": file_path,
        "steps_completed": [],
        "errors": [],
        "refinement_attempts": 0,
        "max_refinement_attempts": 3
    }
    
    try:
        # STEP 1: File Analysis
        print("STEP 1: Analyzing file structure and content...")
        step1_result = analyze_file.invoke({"file_path": file_path})
        if step1_result.startswith("Error"):
            workflow_results["errors"].append(f"Step 1: {step1_result}")
            return json.dumps(workflow_results, indent=2)
        
        workflow_results["steps_completed"].append("file_analysis")
        workflow_results["step1_file_analysis"] = step1_result[:1000] + "..." if len(step1_result) > 1000 else step1_result
        
        # STEP 2: RAG Table Matching
        print("STEP 2: Finding matching database tables...")
        step2_result = find_matching_database_tables.invoke({
            "file_path": file_path, 
            "user_context": user_context or ""
        })
        if step2_result.startswith("Error"):
            workflow_results["errors"].append(f"Step 2: {step2_result}")
            return json.dumps(workflow_results, indent=2)
            
        workflow_results["steps_completed"].append("rag_matching")
        
        # Add better error handling for JSON parsing
        try:
            workflow_results["step2_rag_results"] = json.loads(step2_result)
        except json.JSONDecodeError as e:
            workflow_results["errors"].append(f"Step 2: JSON parsing error - {str(e)}")
            workflow_results["step2_rag_results_raw"] = step2_result[:1000] + "..." if len(step2_result) > 1000 else step2_result
            return json.dumps(workflow_results, indent=2)

        # STEP 3 & 4: Table Selection and Validation Loop
        refinement_history = []
        final_validation = None
        current_step3_result = None
        
        while workflow_results["refinement_attempts"] <= workflow_results["max_refinement_attempts"]:
            # STEP 3: Intelligent Table Selection with Mapping
            attempt_num = workflow_results["refinement_attempts"] + 1
            print(f"STEP 3 (Attempt {attempt_num}): Selecting optimal table and creating field mappings...")
            
            # Build user preferences for this attempt
            user_prefs = []
            if table_name_preference:
                user_prefs.append(f"Table name preference: {table_name_preference}")
            
            # Add refinement guidance from previous validation
            if refinement_history:
                last_validation = refinement_history[-1]["validation"]
                if last_validation.get("refinement_suggestions"):
                    print(f"  Applying refinement suggestions from previous attempt:")
                    for suggestion in last_validation["refinement_suggestions"][:3]:  # Show top 3
                        print(f"    - {suggestion}")
                    user_prefs.extend(last_validation["refinement_suggestions"])
            
            step3_result = intelligent_table_selector.invoke({
                "file_analysis": step1_result, 
                "rag_results": step2_result, 
                "user_preferences": " | ".join(user_prefs) if user_prefs else ""
            })
            
            if step3_result.startswith("Error"):
                workflow_results["errors"].append(f"Step 3 (Attempt {attempt_num}): {step3_result}")
                break
            
            current_step3_result = step3_result
            
            # Extract table selection info for logging
            try:
                step3_data = json.loads(step3_result)
                selected_table = step3_data.get("selected_table", {})
                table_name = selected_table.get("table_name", "Unknown")
                table_confidence = selected_table.get("confidence_score", 0.0)
                field_mappings_count = len(step3_data.get("field_mappings", []))
                unmapped_fields_count = len(step3_data.get("unmapped_fields", []))
                
                print(f"  Selected table: {table_name} (confidence: {table_confidence:.2f})")
                print(f"  Field mappings: {field_mappings_count} mapped, {unmapped_fields_count} unmapped")
            except:
                print(f"  Table selection completed (details in validation)")
            
            # STEP 4: Validation Analysis with Enhanced Logging
            print(f"STEP 4 (Attempt {attempt_num}): Validating mapping quality...")
            print(f"  Starting validation analysis...")
            
            step4_result = results_analyzer.invoke({
                "step3_results": step3_result, 
                "rag_results": step2_result, 
                "table_to_ingest": table_name_preference
            })
            
            if step4_result.startswith("Error"):
                print(f"  Validation failed: {step4_result}")
                workflow_results["errors"].append(f"Step 4 (Attempt {attempt_num}): {step4_result}")
                break
            
            try:
                validation_data = json.loads(step4_result)
                final_validation = validation_data
                
                # Enhanced validation logging
                recommendation = validation_data.get("recommendation")
                confidence = validation_data.get("confidence", 0.0)
                reasoning = validation_data.get("reasoning", "No reasoning provided")
                analysis = validation_data.get("analysis", {})
                
                print(f"  VALIDATION RESULTS:")
                print(f"    Recommendation: {recommendation.upper()}")
                print(f"    Validation confidence: {confidence:.2f}")
                print(f"    Reasoning: {reasoning}")
                
                # Log detailed analysis metrics
                mapping_quality = analysis.get("mapping_quality", {})
                table_selection_analysis = analysis.get("table_selection", {})
                validation_checks = analysis.get("validation_results", {})
                
                print(f"  QUALITY METRICS:")
                print(f"    Mapping confidence: {mapping_quality.get('confidence_score', 0.0):.2f}")
                print(f"    Field coverage: {mapping_quality.get('field_coverage', 0.0):.1%}")
                print(f"    High-confidence mappings: {mapping_quality.get('high_confidence_mappings', 0)}/{mapping_quality.get('mapped_fields', 0)}")
                print(f"    Table rank: #{table_selection_analysis.get('table_rank', 'Unknown')}")
                print(f"    Matches RAG recommendation: {table_selection_analysis.get('matches_recommendation', False)}")
                
                # Log validation issues found
                critical_issues = analysis.get("critical_issues_count", 0)
                warning_issues = analysis.get("warning_issues_count", 0)
                
                if critical_issues > 0 or warning_issues > 0:
                    print(f"  ISSUES DETECTED:")
                    print(f"    Critical issues: {critical_issues}")
                    print(f"    Warning issues: {warning_issues}")
                    
                    # Log specific validation checks that failed
                    failed_checks = [check for check, result in validation_checks.items() if result]
                    if failed_checks:
                        print(f"    Failed checks: {', '.join(failed_checks[:5])}")  # Show first 5
                
                # Log refinement suggestions if any
                refinement_suggestions = validation_data.get("refinement_suggestions", [])
                if refinement_suggestions:
                    print(f"  REFINEMENT SUGGESTIONS:")
                    for i, suggestion in enumerate(refinement_suggestions[:3], 1):  # Show top 3
                        print(f"    {i}. {suggestion}")
                
                # Record this refinement attempt
                refinement_history.append({
                    "attempt": attempt_num,
                    "table_selection": json.loads(step3_result) if not step3_result.startswith("Error") else step3_result,
                    "validation": validation_data,
                    "recommendation": recommendation,
                    "confidence": confidence
                })
                
                print(f"  DECISION: {recommendation} (confidence: {confidence:.2f})")
                
                # Decision logic based on validation recommendation
                if recommendation == "proceed_ingestion":
                    print("  Validation passed - proceeding with ingestion")
                    workflow_results["workflow_status"] = "validation_passed"
                    break
                elif recommendation == "manual_review":
                    print("  Manual review required - stopping automatic refinement")
                    print(f"    Reason: {critical_issues} critical issues detected")
                    workflow_results["workflow_status"] = "requires_manual_review"
                    break
                elif recommendation == "refine_mapping":
                    if workflow_results["refinement_attempts"] < workflow_results["max_refinement_attempts"]:
                        remaining_attempts = workflow_results["max_refinement_attempts"] - workflow_results["refinement_attempts"]
                        print(f"  Refinement recommended - attempting refinement {workflow_results['refinement_attempts'] + 1}")
                        print(f"    Remaining attempts: {remaining_attempts}")
                        workflow_results["refinement_attempts"] += 1
                        continue
                    else:
                        print("  Max refinement attempts reached - requiring manual review")
                        print(f"    Completed {workflow_results['max_refinement_attempts']} refinement attempts")
                        workflow_results["workflow_status"] = "max_refinements_reached"
                        break
                else:
                    print(f"  Unknown validation recommendation: {recommendation}")
                    workflow_results["workflow_status"] = "unknown_validation_result"
                    break
                    
            except json.JSONDecodeError as e:
                print(f"  Failed to parse validation results: {str(e)}")
                workflow_results["errors"].append(f"Step 4 validation parsing error: {str(e)}")
                break
        
        # Record final results
        if current_step3_result:
            workflow_results["steps_completed"].extend(["table_selection_and_mapping", "validation_analysis"])
            try:
                workflow_results["step3_table_selection"] = json.loads(current_step3_result)
            except:
                workflow_results["step3_table_selection"] = current_step3_result
        
        if final_validation:
            workflow_results["step4_validation"] = final_validation
        
        workflow_results["refinement_history"] = refinement_history
        
        # Generate comprehensive mapping visualization for final result
        if current_step3_result and not current_step3_result.startswith("Error"):
            print("Generating comprehensive structure mapping visualization...")
            mapping_visualization = generate_mapping_visualization.invoke({
                "rag_results": json.dumps(workflow_results["step2_rag_results"]),
                "table_selection": current_step3_result,
                "file_analysis": workflow_results["step1_file_analysis"]
            })
            workflow_results["mapping_visualization"] = json.loads(mapping_visualization)
        
        # Set final workflow status and execution readiness
        if workflow_results.get("workflow_status") == "validation_passed":
            workflow_results["ready_for_execution"] = True
            workflow_results["ready_for_review"] = True
            print("All 4 steps completed successfully! Mapping validated and ready for execution.")
        elif workflow_results.get("workflow_status") in ["requires_manual_review", "max_refinements_reached"]:
            workflow_results["ready_for_execution"] = False
            workflow_results["ready_for_review"] = True
            print("Workflow completed but requires human review before execution.")
        else:
            workflow_results["ready_for_execution"] = False
            workflow_results["ready_for_review"] = False
            workflow_results["workflow_status"] = workflow_results.get("workflow_status", "completed_with_issues")
        
        return json.dumps(workflow_results, indent=2)
        
    except Exception as e:
        workflow_results["workflow_status"] = "failed"
        workflow_results["errors"].append(f"Orchestrator error: {str(e)}")
        return json.dumps(workflow_results, indent=2)

@tool 
def results_analyzer(step3_results: str, rag_results: str, table_to_ingest: Optional[str] = None) -> str:
    """ 
    VALIDATION AGENT: Analyze the table mapping results and provide intelligent recommendations.
    
    This agent performs comprehensive validation of the mapping quality and determines
    whether to proceed with ingestion, refine the mapping, or require manual review.

    Args:
        step3_results: JSON string from intelligent_table_selector (Step 3)
        rag_results: JSON string from find_matching_database_tables (Step 2)
        table_to_ingest: Optional original table preference for comparison
    
    Returns: 
        JSON with recommendation and detailed analysis: 
        {
            "recommendation": "proceed_ingestion" | "refine_mapping" | "manual_review",
            "confidence": 0.0-1.0,
            "analysis": {...},
            "refinement_suggestions": [...],
            "reasoning": "detailed explanation"
        }
    """
    
    try:
        # Parse input data
        results = json.loads(step3_results)
        rag_data = json.loads(rag_results)
        
        # Extract key information
        selected_table = results["selected_table"]
        confidence_score = selected_table["confidence_score"] 
        chosen_table_name = selected_table["table_name"]
        
        field_mappings = results.get("field_mappings", [])
        unmapped_fields = results.get("unmapped_fields", [])
        data_quality_concerns = results.get("data_quality_concerns", [])
        estimated_success_rate = results.get("estimated_success_rate", 0.0)
        
        top_tables = rag_data.get("top_10_tables", [])
        recommended_table = rag_data.get("recommended_table", "")
        file_info = rag_data.get("file_analysis", {})
        
        # Calculate validation metrics
        total_fields = len(field_mappings) + len(unmapped_fields)
        mapping_coverage = len(field_mappings) / max(total_fields, 1)
        
        # Field mapping quality analysis
        high_confidence_mappings = sum(1 for m in field_mappings if m.get("confidence", 0) >= 0.8)
        low_confidence_mappings = sum(1 for m in field_mappings if m.get("confidence", 0) < 0.6)
        complex_transformations = sum(1 for m in field_mappings if m.get("transformation", "none") != "none")
        
        # Alternative table analysis
        better_alternatives = []
        table_rank = None
        for i, table in enumerate(top_tables):
            if table.get("table_name") == chosen_table_name:
                table_rank = i + 1
            elif table.get("composite_score", 0) > confidence_score + 0.1:
                better_alternatives.append({
                    "name": table.get("table_name"),
                    "score": table.get("composite_score", 0),
                    "rank": i + 1
                })
        
        # Validation criteria evaluation
        validation_checks = {
            "confidence_too_low": confidence_score < 0.75,
            "poor_field_coverage": mapping_coverage < 0.7,
            "high_unmapped_ratio": len(unmapped_fields) / max(total_fields, 1) > 0.3,
            "data_quality_issues": len(data_quality_concerns) > 0,
            "low_success_rate": estimated_success_rate < 0.7,
            "many_low_confidence_mappings": low_confidence_mappings > len(field_mappings) * 0.3,
            "complex_transformations": complex_transformations > len(field_mappings) * 0.4,
            "better_alternatives_exist": len(better_alternatives) > 0,
            "not_recommended_table": chosen_table_name != recommended_table and recommended_table != "",
            "low_table_rank": table_rank is not None and table_rank > 3
        }
        
        # Count critical issues
        critical_issues = sum([
            validation_checks["confidence_too_low"],
            validation_checks["poor_field_coverage"],
            validation_checks["data_quality_issues"],
            validation_checks["low_success_rate"]
        ])
        
        warning_issues = sum([
            validation_checks["high_unmapped_ratio"],
            validation_checks["many_low_confidence_mappings"],
            validation_checks["complex_transformations"],
            validation_checks["not_recommended_table"]
        ])
        
        # Decision logic with detailed reasoning
        reasoning_parts = []
        refinement_suggestions = []
        
        # CRITICAL ISSUES - Manual Review Required
        if critical_issues >= 2:
            recommendation = "manual_review"
            analysis_confidence = 0.9
            reasoning_parts.append(f"Multiple critical issues detected ({critical_issues}/4)")
            
            if validation_checks["confidence_too_low"]:
                reasoning_parts.append(f"• Low mapping confidence: {confidence_score:.2f} < 0.75")
                refinement_suggestions.append("Consider alternative tables with higher confidence scores")
            
            if validation_checks["poor_field_coverage"]:
                reasoning_parts.append(f"• Poor field coverage: {mapping_coverage:.1%} < 70%")
                refinement_suggestions.append("Review unmapped fields and consider schema modifications")
            
            if validation_checks["data_quality_issues"]:
                reasoning_parts.append(f"• Data quality concerns: {len(data_quality_concerns)} issues")
                refinement_suggestions.append("Address data quality issues before proceeding")
            
            if validation_checks["low_success_rate"]:
                reasoning_parts.append(f"• Low estimated success rate: {estimated_success_rate:.1%} < 70%")
                refinement_suggestions.append("Investigate causes of low success rate prediction")
        
        # REFINEMENT OPPORTUNITY - Better alternatives exist
        elif validation_checks["better_alternatives_exist"] and validation_checks["confidence_too_low"]:
            recommendation = "refine_mapping"
            analysis_confidence = 0.85
            reasoning_parts.append("Better table alternatives available with higher confidence")
            reasoning_parts.append(f"Current choice: {chosen_table_name} (confidence: {confidence_score:.2f})")
            
            for alt in better_alternatives[:2]:  # Show top 2 alternatives
                reasoning_parts.append(f"• Alternative: {alt['name']} (score: {alt['score']:.2f}, rank: {alt['rank']})")
                refinement_suggestions.append(f"Try mapping to {alt['name']} table instead")
        
        # WARNING ISSUES - Conditional refinement
        elif warning_issues >= 2 or (warning_issues >= 1 and validation_checks["not_recommended_table"]):
            recommendation = "refine_mapping"
            analysis_confidence = 0.75
            reasoning_parts.append(f"Multiple warning indicators suggest refinement ({warning_issues} warnings)")
            
            if validation_checks["high_unmapped_ratio"]:
                unmapped_ratio = len(unmapped_fields) / max(total_fields, 1)
                reasoning_parts.append(f"• High unmapped field ratio: {unmapped_ratio:.1%}")
                refinement_suggestions.append("Try alternative tables that might map more fields")
            
            if validation_checks["many_low_confidence_mappings"]:
                reasoning_parts.append(f"• Many low-confidence mappings: {low_confidence_mappings}/{len(field_mappings)}")
                refinement_suggestions.append("Refine search queries to find better field matches")
            
            if validation_checks["not_recommended_table"]:
                reasoning_parts.append(f"• Chosen table differs from RAG recommendation: {recommended_table}")
                refinement_suggestions.append(f"Consider using RAG-recommended table: {recommended_table}")
        
        # GOOD QUALITY - Proceed with ingestion
        else:
            recommendation = "proceed_ingestion"
            analysis_confidence = 0.9
            reasoning_parts.append("Mapping quality meets acceptance criteria")
            reasoning_parts.append(f"• Confidence score: {confidence_score:.2f} (acceptable)")
            reasoning_parts.append(f"• Field coverage: {mapping_coverage:.1%} (good)")
            reasoning_parts.append(f"• Success rate estimate: {estimated_success_rate:.1%} (acceptable)")
            
            if high_confidence_mappings > 0:
                reasoning_parts.append(f"• High-confidence mappings: {high_confidence_mappings}/{len(field_mappings)}")
        
        # Compile detailed analysis
        analysis = {
            "mapping_quality": {
                "confidence_score": confidence_score,
                "field_coverage": mapping_coverage,
                "total_fields": total_fields,
                "mapped_fields": len(field_mappings),
                "unmapped_fields": len(unmapped_fields),
                "high_confidence_mappings": high_confidence_mappings,
                "low_confidence_mappings": low_confidence_mappings,
                "complex_transformations": complex_transformations
            },
            "table_selection": {
                "chosen_table": chosen_table_name,
                "table_rank": table_rank,
                "recommended_table": recommended_table,
                "matches_recommendation": chosen_table_name == recommended_table,
                "alternatives_available": len(better_alternatives)
            },
            "validation_results": validation_checks,
            "critical_issues_count": critical_issues,
            "warning_issues_count": warning_issues,
            "data_quality_concerns": data_quality_concerns,
            "estimated_success_rate": estimated_success_rate
        }
        
        # Enhanced refinement suggestions based on specific issues
        if recommendation == "refine_mapping":
            if validation_checks["better_alternatives_exist"]:
                refinement_suggestions.append("Re-run intelligent_table_selector with alternative table preference")
            
            if validation_checks["poor_field_coverage"]:
                refinement_suggestions.append("Expand search queries to include more domain-specific terms")
            
            if validation_checks["many_low_confidence_mappings"]:
                refinement_suggestions.append("Add user context about field meanings and business logic")
        
        return json.dumps({
            "recommendation": recommendation,
            "confidence": analysis_confidence,
            "analysis": analysis,
            "refinement_suggestions": refinement_suggestions,
            "reasoning": " | ".join(reasoning_parts),
            "next_action": _get_next_action_instructions(recommendation, refinement_suggestions),
            "validation_timestamp": "2025-09-29",
            "requires_human_input": recommendation == "manual_review"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "recommendation": "manual_review",
            "confidence": 0.0,
            "analysis": {"error": str(e)},
            "refinement_suggestions": ["Fix validation agent error and retry"],
            "reasoning": f"Validation agent encountered error: {str(e)}",
            "next_action": "Debug validation agent and retry analysis",
            "requires_human_input": True
        }, indent=2)

def _get_next_action_instructions(recommendation: str, suggestions: list) -> str:
    """Generate specific next action instructions based on recommendation."""
    if recommendation == "proceed_ingestion":
        return "Execute database ingestion with current mapping configuration"
    elif recommendation == "refine_mapping":
        if suggestions:
            return f"Implement refinement suggestions: {suggestions[0]}"
        return "Re-run table selection with enhanced search parameters"
    else:  # manual_review
        return "Escalate to human review - automatic refinement not recommended"

# Available tools for the agent
tools_analyze = [
    analyze_file,
    python_code_executor
]

tools_retriever = [
    find_matching_database_tables,
    entity_first_database_discovery
]

tools_workflow = [
    intelligent_table_selector,
    database_ingestion_orchestrator,
    generate_mapping_visualization
]

# Complete tool set for the enhanced single agent
all_ingestion_tools = tools_analyze + tools_retriever + tools_workflow + [results_analyzer]