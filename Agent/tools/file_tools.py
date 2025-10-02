import os
import base64
import pandas as pd
import numpy as np
import json
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

PDF_AVAILABLE = PDFPLUMBER_AVAILABLE and PYMUPDF_AVAILABLE


def _get_vision_llm() -> ChatOpenAI:
    """
    Initialize a vision-capable language model for OCR tasks.

    Returns: 
        ChatOpenAI: An instance of ChatOpenAI configured for vision tasks.
    """
    from ..config.settings import get_settings
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=settings.temperature
    )


@tool
def analyze_file(file_path: str) -> str:
    """
    Analyze various file types including PDF, CSV, Excel, JSON, images, and text files.
    Extracts content and provides structured analysis for database ingestion.
    
    Args:
        file_path: Path to the file to analyze
    """

    vision_llm = _get_vision_llm()
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if (file_extension == '.pdf'):
            return _analyze_pdf(file_path, vision_llm)
        elif file_extension in ['.csv']:
            return _analyze_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return _analyze_excel(file_path)
        elif file_extension == '.json':
            return _analyze_json(file_path)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            return _analyze_image(file_path, vision_llm)
        elif file_extension in ['.txt', '.xml']:
            return _analyze_text(file_path)
        else:
            return f"Unsupported file type: {file_extension}"
    
    except Exception as e:
        return f"Error analyzing file: {str(e)}"

def _analyze_pdf(file_path: str, vision_llm: ChatOpenAI) -> str:
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
                ocr_text = _extract_text_from_image_base64(image_base64, vision_llm)
                if ocr_text:
                    ocr_pages.append(f"Page {page_num+1} (OCR):\n{ocr_text}")
                
                elif ocr_text.startswith("Error"):
                    print(f"OCR failed for page {page_num+1}: {ocr_text}")
            
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
    
    class NumpyEncoder(json.JSONEncoder):
        """Handle numpy types in JSON serialization."""
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return super().default(obj)
    
    try:
        # Add delimiter detection for semicolon-separated files
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        
        delimiter = '\t' if '\t' in first_line else ',' if ',' in first_line else ';'
        df = pd.read_csv(file_path, delimiter=delimiter)

        data_type_list = []
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            # Infer general data type category for mapping
            if dtype == 'object':
                max_length = int(df[col].astype(str).str.len().max())
                data_type_category = "Text"
            elif dtype in ['int64', 'int32']:
                max_length = None
                data_type_category = "Integer"
            elif dtype in ['float64', 'float32']:
                max_length = None
                data_type_category = "Decimal"
            elif 'datetime' in str(dtype):
                max_length = None
                data_type_category = "DateTime"
            else:
                max_length = None
                data_type_category = "Unknown"
            
            # Convert sample values to native Python types
            sample_values = [
                None if pd.isna(val) else 
                int(val) if isinstance(val, (np.integer, np.int64, np.int32)) else
                float(val) if isinstance(val, (np.floating, np.float64, np.float32)) else
                val
                for val in df[col].dropna().head(3).tolist()
            ]

            data_type_list.append({
                "name": col,
                "data_type": data_type_category,
                "max_length": max_length,
                "null_count": int(null_count),
                "unique_count": int(unique_count),
                "sample_values": sample_values
            })
        
        return json.dumps({
            "file_type": "csv",
            "file_name": os.path.basename(file_path),
            "dimensions": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1])
            },
            "columns": list(df.columns),
            "data_types": data_type_list,
            "sample_data": df.head(5).to_dict('records'),
            "delimiter": delimiter, 
            "encoding": "utf-8"
        }, cls=NumpyEncoder)  # Use custom encoder for all numpy types
    
    except Exception as e:
        return json.dumps({
            "file_type": "csv",
            "error": str(e),
            "analysis_success": False
        })

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

def _analyze_image(file_path: str, vision_llm: ChatOpenAI) -> str:
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

def _extract_text_from_image_base64(image_base64: str, vision_llm: ChatOpenAI) -> str:
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
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

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
