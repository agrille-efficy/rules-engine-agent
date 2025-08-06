import os
import pandas as pd
import json
import base64
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image
import pytesseract
from dotenv import load_dotenv

# Import SQLCodeParser from app.py
from app import SQLCodeParser

# PDF processing imports
try:
    import PyPDF2
    import pdfplumber
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

load_dotenv()
vision_llm = ChatOpenAI(model="gpt-4o")

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
        if file_extension == '.pdf':
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
    
    # If no text found, try OCR on PDF images
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
    """Analyze CSV files for database schema generation."""
    try:
        df = pd.read_csv(file_path)
        
        results = []
        results.append(f"CSV Analysis for: {os.path.basename(file_path)}")
        results.append(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        results.append(f"Columns: {list(df.columns)}")
        
        # Data types analysis
        results.append("\nData Types Analysis:")
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            # Infer SQL data type
            if dtype == 'object':
                max_length = df[col].astype(str).str.len().max()
                sql_type = f"VARCHAR({min(max_length + 50, 500)})"
            elif dtype in ['int64', 'int32']:
                sql_type = "INTEGER"
            elif dtype in ['float64', 'float32']:
                sql_type = "DECIMAL(10,2)"
            elif 'datetime' in str(dtype):
                sql_type = "DATETIME"
            else:
                sql_type = "TEXT"
            
            results.append(f"  {col}: {sql_type} (nulls: {null_count}, unique: {unique_count})")
        
        # Sample data
        results.append(f"\nSample Data (first 3 rows):")
        results.append(df.head(3).to_string())
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"

def _analyze_excel(file_path: str) -> str:
    """Analyze Excel files for database schema generation."""
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
                    if dtype == 'object':
                        max_length = df[col].astype(str).str.len().max()
                        sql_type = f"VARCHAR({min(max_length + 50, 500)})"
                    elif dtype in ['int64', 'int32']:
                        sql_type = "INTEGER"
                    elif dtype in ['float64', 'float32']:
                        sql_type = "DECIMAL(10,2)"
                    else:
                        sql_type = "TEXT"
                    
                    results.append(f"  {col}: {sql_type}")
                
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
        
        response = vision_llm(message)
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
        
        response = vision_llm(message)
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
def generate_sql_schema(analysis_result: str, table_name: str = None) -> str:
    """
    Generate CREATE TABLE and INSERT SQL statements based on file analysis.
    
    Args:
        analysis_result: The result from analyze_file function
        table_name: Optional custom table name
    """
    try:
        # Use vision model to generate SQL from analysis
        prompt = f"""Based on the following file analysis, generate ONLY SQL statements for database ingestion:

{analysis_result}

Requirements:
- CREATE TABLE statement with appropriate data types for table: {table_name if table_name else 'extracted_data'}
- 2-3 sample INSERT statements
- Use SQL data types: VARCHAR, INTEGER, DECIMAL, DATETIME, TEXT, etc.
- Include NOT NULL constraints where appropriate
- Create meaningful column names

IMPORTANT: Return ONLY the SQL code. No explanations, no markdown formatting, no code blocks. Start directly with CREATE TABLE and end with the last INSERT statement."""
        
        response = vision_llm([HumanMessage(content=prompt)])
        
        # Use the robust SQLCodeParser to clean the response
        clean_sql = SQLCodeParser.extract_sql_code(response.content)
        
        return clean_sql
    
    except Exception as e:
        return f"Error generating SQL: {str(e)}"

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

# Available tools for the agent
tools = [
    analyze_file,
    generate_sql_schema,
    python_code_executor
]