import os
import sys
import gradio as gr
import tempfile
import pandas as pd
import json
import base64
import re
from typing import Optional, Tuple
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(r'C:\Users\axel.grille\Documents\rules-engine-agent\Agent\.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PDF processing imports
try:
    import PyPDF2
    import pdfplumber
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

class SQLCodeParser:
    """A utility class to parse and extract SQL code from LLM responses."""
    
    @staticmethod
    def extract_sql_code(response_text: str) -> str:
        """
        Extract clean SQL code from LLM response text.
        
        Args:
            response_text: The full response text from the LLM containing SQL
            
        Returns:
            Clean SQL code without markdown formatting or explanatory text
        """
        try:
            # Extract from ```sql code blocks
            sql_blocks = re.findall(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
            
            if sql_blocks:
                # Join all SQL blocks found
                sql_code = '\n\n'.join(sql_blocks)
            else:
                #Extract from ``` code blocks (without language specification)
                code_blocks = re.findall(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                sql_code = '\n\n'.join(code_blocks) if code_blocks else response_text
            

            # Clean up the extracted SQL
            cleaned_sql = SQLCodeParser._clean_sql_code(sql_code)
            
            return cleaned_sql
            
        except Exception as e:
            return f"-- Error parsing SQL: {str(e)}\n{response_text}"
    
    @staticmethod
    def _clean_sql_code(sql_code: str) -> str:
        """
        Clean and format the extracted SQL code.
        
        Args:
            sql_code: Raw SQL code string
            
        Returns:
            Cleaned and formatted SQL code
        """
        # Remove common markdown artifacts
        sql_code = re.sub(r'^```sql\s*', '', sql_code, flags=re.MULTILINE | re.IGNORECASE)
        sql_code = re.sub(r'^```\s*$', '', sql_code, flags=re.MULTILINE)
        
        # Remove comment lines that are not SQL comments
        lines = sql_code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped and not cleaned_lines:
                continue
                
            # Always keep SQL comments
            if stripped.startswith('--') or stripped.startswith('/*') or '*/' in stripped:
                cleaned_lines.append(line)
                continue

            # Skip empty lines
            if not stripped:
                continue
                
            # Check if line starts with SQL keywords
            if any(stripped.upper().startswith(keyword) for keyword in 
                    ['CREATE', 'INSERT', 'UPDATE', 'DELETE', 'SELECT', 'ALTER', 'DROP']):
                cleaned_lines.append(line)
                
            # Keep lines that are part of multi-line SQL statements
            elif stripped and (stripped.endswith(',') or stripped.endswith('(') or 
                             stripped.startswith(')') or 'VALUES' in stripped.upper() or
                             any(keyword in stripped.upper() for keyword in 
                                 ['PRIMARY KEY', 'FOREIGN KEY', 'NOT NULL', 'DEFAULT', 'CHECK'])):
                cleaned_lines.append(line)

            # Keep statement terminators and closing brackets
            elif stripped in [');', ')', ';'] or stripped.endswith(');'):
                cleaned_lines.append(line)
                
            # Keep lines that look like column definitions or table content
            elif '(' in stripped or ')' in stripped or ',' in stripped:
                cleaned_lines.append(line)

        result = '\n'.join(cleaned_lines)
        
        # Remove excessive blank lines
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        result = result.strip()
        
        return result
    
    @staticmethod
    def separate_statements(sql_code: str) -> dict:
        """
        Separate CREATE TABLE and INSERT statements.
        
        Args:
            sql_code: Complete SQL code string
            
        Returns:
            Dictionary with 'create_table' and 'insert_statements' keys
        """
        try:
            # Split by semicolon but be careful with semicolons in strings
            statements = []
            current_statement = ""
            in_string = False
            quote_char = None
            
            for char in sql_code:
                if char in ['"', "'"] and not in_string:
                    in_string = True
                    quote_char = char
                elif char == quote_char and in_string:
                    in_string = False
                    quote_char = None
                elif char == ';' and not in_string:
                    if current_statement.strip():
                        statements.append(current_statement.strip())
                    current_statement = ""
                    continue
                
                current_statement += char
            
            # Add the last statement if it doesn't end with semicolon
            if current_statement.strip():
                statements.append(current_statement.strip())
            
            # Categorize statements
            create_tables = []
            insert_statements = []
            
            for statement in statements:
                statement_upper = statement.strip().upper()
                if statement_upper.startswith('CREATE TABLE'):
                    create_tables.append(statement)
                elif statement_upper.startswith('INSERT INTO'):
                    insert_statements.append(statement)
            
            return {
                'create_table': '\n\n'.join(create_tables),
                'insert_statements': '\n\n'.join(insert_statements),
                'all_statements': statements
            }
            
        except Exception as e:
            return {
                'create_table': f"-- Error separating statements: {str(e)}",
                'insert_statements': "",
                'all_statements': [sql_code]
            }

class StandaloneFileProcessor:
    """A standalone file processor without LangChain tool schemas."""
    
    def __init__(self):
        """Initialize the processor"""
        self.vision_llm = ChatOpenAI(model="gpt-4o") if OPENAI_API_KEY else None
        print("StandaloneFileProcessor initialized.")

    def analyze_file(self, file_path: str) -> str:
        """Analyze various file types"""
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                return self._analyze_pdf(file_path)
            elif file_extension == '.csv':
                return self._analyze_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return self._analyze_excel(file_path)
            elif file_extension == '.json':
                return self._analyze_json(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                return self._analyze_image(file_path)
            elif file_extension in ['.txt', '.xml']:
                return self._analyze_text(file_path)
            else:
                return f"Error: Unsupported file type: {file_extension}"
        except Exception as e:
            return f"Error: Error analyzing file: {str(e)}"

    def _analyze_pdf(self, file_path: str) -> str:
        """Analyze PDF files"""
        if not PDF_AVAILABLE:
            return "Error: PDF processing libraries not installed."
        
        results = []
        text_content = ""
        
        try:
            with pdfplumber.open(file_path) as pdf:
                text_pages = []
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(f"Page {i+1}:\n{page_text[:1000]}")
                
                if text_pages:
                    text_content = "\n\n".join(text_pages)
        except Exception as e:
            results.append(f"Text extraction error: {str(e)}")
        
        if text_content:
            results.append(f"Structural Analysis:")
            
            # Analyze structure
            lines = text_content.split('\n')
            potential_headers = [line.strip() for line in lines[:20] if line.strip() and len(line.strip()) < 100]
            
            results.append(f"- Total content length: {len(text_content)} characters")
            results.append(f"- Potential headers/fields: {potential_headers[:5]}")
            
            # Look for tabular data
            table_patterns = [line.strip() for line in lines if '|' in line or '\t' in line or ',' in line]
            if table_patterns:
                results.append(f"- Potential tabular data: {len(table_patterns)} lines")
        else:
            results.append("Error: No text content could be extracted from the PDF")
        
        return '\n'.join(results)

    def _analyze_csv(self, file_path: str) -> str:
        """Analyze CSV files"""
        try:
            df = pd.read_csv(file_path)
            results = []
            results.append(f"CSV Analysis for: {os.path.basename(file_path)}")
            results.append(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
            results.append(f"Columns: {list(df.columns)}")
            
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
            
            results.append(f"\nSample Data (first 3 rows):")
            results.append(df.head(3).to_string())
            
            return '\n'.join(results)
        except Exception as e:
            return f"Error: Error analyzing CSV: {str(e)}"

    def _analyze_excel(self, file_path: str) -> str:
        """Analyze Excel files"""
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            results = []
            results.append(f"Excel Analysis for: {os.path.basename(file_path)}")
            results.append(f"Sheets found: {sheet_names}")
            
            # Analyze first sheet
            if sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_names[0])
                results.append(f"\n--- Sheet: {sheet_names[0]} ---")
                results.append(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
                results.append(f"Columns: {list(df.columns)}")
                
                results.append(f"\nSample Data:")
                results.append(df.head(3).to_string())
            
            return '\n'.join(results)
        except Exception as e:
            return f"Error: Error analyzing Excel: {str(e)}"

    def _analyze_json(self, file_path: str) -> str:
        """Analyze JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = []
            results.append(f"JSON Analysis for: {os.path.basename(file_path)}")
            
            if isinstance(data, list):
                results.append(f"Type: Array with {len(data)} items")
            elif isinstance(data, dict):
                results.append(f"Type: Object with {len(data)} keys")
                results.append(f"Keys: {list(data.keys())}")
            
            results.append(f"\nStructure Preview:")
            preview = json.dumps(data, indent=2)[:1000]
            results.append(preview + "..." if len(str(data)) > 1000 else preview)
            
            return '\n'.join(results)
        except Exception as e:
            return f"Error: Error analyzing JSON: {str(e)}"

    def _analyze_image(self, file_path: str) -> str:
        """Analyze images using vision model."""
        if not self.vision_llm:
            return "Error: Vision model not available (OpenAI API key required)"
        
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
                                "Provide the extracted information in a clear, structured format suitable for database storage."
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
            
            response = self.vision_llm.invoke(message)
            return f"Image Analysis for: {os.path.basename(file_path)}\n\n{response.content}"
        except Exception as e:
            return f"Error: Error analyzing image: {str(e)}"

    def _analyze_text(self, file_path: str) -> str:
        """Analyze text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            results = []
            results.append(f"Text Analysis for: {os.path.basename(file_path)}")
            results.append(f"File size: {len(content)} characters")
            results.append(f"Lines: {len(content.split(chr(10)))}")
            results.append(f"\nContent Preview:")
            results.append(content[:1000] + "..." if len(content) > 1000 else content)
            
            return '\n'.join(results)
        except Exception as e:
            return f"Error: Error analyzing text file: {str(e)}"

    def generate_sql_schema(self, analysis_result: str, table_name: str = "extracted_data") -> str:
        """Generate SQL schema from analysis."""
        if not self.vision_llm:
            return "Error: SQL generation requires OpenAI API key"
        
        try:
            prompt = f"""
            Based on the following file analysis, generate SQL statements for database ingestion:

            {analysis_result}
            
            Please provide:
            1. A CREATE TABLE statement with appropriate data types
            2. Sample INSERT statements
            3. Use table name: {table_name}
            
            Make sure to:
            - Use appropriate SQL data types (VARCHAR, INTEGER, DECIMAL, DATETIME, etc.)
            - Include NOT NULL constraints where appropriate
            - Create meaningful column names
            - Provide at least 2-3 sample INSERT statements
            """
            
            response = self.vision_llm.invoke([HumanMessage(content=prompt)])

            # Parse the SQL code from the response
            clean_sql = SQLCodeParser.extract_sql_code(response.content)
            
            # Separate statements if needed
            separated = SQLCodeParser.separate_statements(clean_sql)
            
            return f"Generated SQL for table '{table_name}':\n\n{clean_sql}"
        except Exception as e:
            return f"Error: Error generating SQL: {str(e)}"


# Global processor instance
processor = None

def initialize_processor():
    """Initialize the file processor"""
    global processor
    if processor is None:
        try:
            processor = StandaloneFileProcessor()
            return "File processor initialized successfully!"
        except Exception as e:
            return f"Error initializing processor: {str(e)}"
    return "Processor already initialized!"

def process_file_upload(file, table_name: Optional[str] = None) -> Tuple[str, str]:
    """Process uploaded file and generate SQL."""
    if file is None:
        return "No file uploaded.", ""
    
    try:
        # Get file path from Gradio file object
        if hasattr(file, 'name'):
            file_path = file.name
        else:
            file_path = str(file)
        
        if not os.path.exists(file_path):
            return f"File not found: {file_path}", ""
        
        # Initialize processor if needed
        if processor is None:
            init_result = initialize_processor()
            if "Error" in init_result:
                return init_result, ""
        
        # Analyze file
        analysis = processor.analyze_file(file_path)
        
        # Generate SQL
        table_name = table_name or "extracted_data"
        sql = processor.generate_sql_schema(analysis, table_name)
        
        return analysis, sql
        
    except Exception as e:
        return f"Error processing file: {str(e)}", ""

def process_question(question: str) -> str:
    """Process a general question."""
    if not question or not question.strip():
        return "Please provide a question."
    
    return f"Question received: {question}\n\nThis is a simplified interface focused on file processing.\n\nFor advanced agent capabilities, use: python app.py --cli"


### GRADIO INTERFACE CREATION ###
def create_gradio_interface():
    """Create a clean Gradio interface."""
    
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .file-upload {
        border: 2px dashed #ccc !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    .output-text {
        font-family: 'Monaco', 'Consolas', monospace !important;
        font-size: 12px !important;
    }
    """
    
    with gr.Blocks(css=css, title="File Analysis & SQL Generation") as interface:
        
        gr.Markdown("""
        # Rules Engine Agent - File Processor
        ### Analyze Files & Generate SQL for Database Ingestion
        
        Upload files (PDF, CSV, Excel, Images, JSON, etc.) to extract structured data and generate SQL queries.
        """)
        
        # Initialization
        with gr.Row():
            init_btn = gr.Button("Initialize Processor", variant="primary")
            init_status = gr.Textbox(label="Status", interactive=False)
            
        init_btn.click(fn=initialize_processor, outputs=init_status)
        
        gr.Markdown("---")
        
        # File processing
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### File Upload")
                
                file_input = gr.File(
                    label="Upload File",
                    file_types=[".csv", ".xlsx", ".xls", ".json", ".txt", ".xml", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp"]
                )
                
                table_name_input = gr.Textbox(
                    label="Table Name (Optional)",
                    placeholder="extracted_data",
                    value=""
                )
                
                analyze_btn = gr.Button("Analyze & Generate SQL", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                
                analysis_output = gr.Textbox(
                    label="File Analysis",
                    lines=15,
                    interactive=False,
                    elem_classes=["output-text"]
                )
                
                sql_output = gr.Textbox(
                    label="SQL Schema",
                    lines=10,
                    interactive=False,
                    elem_classes=["output-text"]
                )
        
        analyze_btn.click(
            fn=process_file_upload,
            inputs=[file_input, table_name_input],
            outputs=[analysis_output, sql_output]
        )
        
        gr.Markdown("---")
        
        with gr.Accordion("Help & Examples", open=False):
            gr.Markdown("""
            ### Supported File Types:
            - **PDF**: Text extraction with OCR fallback
            - **CSV/Excel**: Data analysis and SQL table generation  
            - **Images**: Document OCR and structured data extraction
            - **JSON**: Structure analysis and schema generation
            - **Text/XML**: Content analysis
            
            ### Features:
            - Automatic data type inference
            - SQL CREATE TABLE generation
            - Sample INSERT statements
            - Column analysis and statistics
            - Image-based document processing
            """)
    
    return interface


def run_agent_cli(question_text: str) -> str:
    """Run the agent on a user-provided question (legacy CLI function)"""
    return process_question(question_text)


def test_simple_question():
    """Test with a simple question that doesn't require external APIs"""
    print("\nTesting with simple math question...")
    question = "What is 15 + 27?"
    answer = run_agent_cli(question)
    print(f"Test result: {answer}")


def main():
    """Main function - launches Gradio interface by default, CLI with --cli flag"""
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        print("\n" + "="*60)
        print("AGENT COMMAND LINE INTERFACE")
        print("="*60)
        print("Welcome! You can ask questions and the agent will help you.")
        print("Type 'test' to run a simple test")
        print("Type 'quit', 'exit', or 'q' to stop the program.")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get user input
                question = input("Ask me anything (or 'test' for simple test): ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print("\nGoodbye! Thanks for using the agent.")
                    break
                
                # Check for test command
                if question.lower() == 'test':
                    test_simple_question()
                    continue
                
                print("\n" + "-"*40)
                print("Agent is thinking...")
                print("-"*40)
                
                # Run the agent
                answer = run_agent_cli(question)
                
                print(f"\nANSWER:\n{answer}\n")
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for using the agent.")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                print("Please try again.\n")
    else:
        # Launch Gradio interface
        print("Launching Gradio GUI interface...")
        print("Use --cli flag to run in command line mode")
        
        try:
            interface = create_gradio_interface()
            interface.launch(
                server_name="127.0.0.1",
                server_port=7860,
                share=True,
                show_error=False,  # Disable error display to avoid schema issues
                debug=False
            )
        except Exception as e:
            print(f"Error launching Gradio interface: {e}")
            print("Trying fallback configuration...")
            
            # Fallback: simpler launch configuration
            try:
                interface = create_gradio_interface()
                interface.launch(share=True, debug=False)
            except Exception as e2:
                print(f"Failed to launch interface: {e2}")
                print("Please try running with --cli flag for command line mode")
                print("Example: python app.py --cli")

if __name__ == "__main__":
    main()