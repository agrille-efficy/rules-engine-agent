import os
import sys
import gradio as gr
import pandas as pd
import json
import base64
import signal
from typing import Tuple
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from utils import SQLCodeParser

try:
    # Preferred package-qualified import
    from Agent import tools as _tools_mod  # type: ignore
except Exception:
    # Ensure parent directory (project root) is on sys.path
    import pathlib, importlib
    parent_dir = pathlib.Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    try:
        _tools_mod = importlib.import_module("Agent.tools")
    except Exception as e:
        raise ImportError(f"Failed to import Agent.tools. Run with 'python -m Agent.app'. Original error: {e}")

# Expose needed symbol(s)
database_ingestion_orchestrator = _tools_mod.database_ingestion_orchestrator

# === Import build_graph robustly (supports `python -m Agent.app` and direct `python Agent/app.py`) ===
try:
    # Preferred: executed as a package module
    from Agent.agent import build_graph
    _USING_PKG_IMPORT = True
except Exception:
    # Fallback: adjust sys.path to include parent dir, then import with package context
    import importlib, pathlib, types
    parent_dir = pathlib.Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    try:
        agent_module = importlib.import_module("Agent.agent")
        build_graph = agent_module.build_graph
        _USING_PKG_IMPORT = True
    except Exception as e:
        print("WARNING: Failed to import Agent.agent with package context. Some features may break.")
        print(f"Import error: {e}")
        # Last resort (may break relative imports inside agent):
        try:
            import agent as _flat_agent
            build_graph = _flat_agent.build_graph
            _USING_PKG_IMPORT = False
        except Exception as e2:
            raise ImportError(f"Cannot import build_graph from Agent.agent or agent: {e2}")

# Load environment variables
load_dotenv(r'C:\Users\axel.grille\Documents\rules-engine-agent\Agent\.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Agent execution timed out")

class EnhancedAgent:
    """Enhanced agent with 4-step workflow capabilities."""
    def __init__(self):
        print("Enhanced Agent with 4-Step Workflow initialized.")
        self.graph = build_graph()
        
    def __call__(self, question: str) -> str:
        print(f"Processing question: {question[:100]}...")
        
        # Set up timeout for Windows (if available)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)  # Increased timeout for workflow processing
        
        try:
            messages = [HumanMessage(content=question)]
            thread_config = {"configurable": {"thread_id": "main_conversation"}}
            messages = self.graph.invoke({"messages": messages}, config=thread_config)
            answer = messages['messages'][-1].content
            
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            return answer
        except TimeoutError:
            return "Agent execution timed out. This might be due to slow API responses or complex workflow processing."
        except Exception as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            raise

class WorkflowFileProcessor:
    """Enhanced file processor using the 4-step workflow system."""
    
    def __init__(self):
        """Initialize the enhanced processor"""
        print("WorkflowFileProcessor initialized with 4-step workflow capabilities.")

    def process_file_with_workflow(self, file_path: str, user_context: str = None, table_name_preference: str = None) -> dict:  # noqa: F811
        """Process file using the complete 4-step workflow orchestrator.
        Always returns a dict with a 'success' boolean.
        """
        def _invoke_orchestrator(path, ctx, tbl_pref):
            # Support tool wrapper or plain function
            orch = database_ingestion_orchestrator
            if hasattr(orch, 'func'):  # LangChain tool wrapper
                return orch.func(file_path=path, user_context=ctx, table_name_preference=tbl_pref)
            elif hasattr(orch, 'invoke'):  # Use invoke method
                return orch.invoke({
                    "file_path": path, 
                    "user_context": ctx, 
                    "table_name_preference": tbl_pref
                })
            return orch(path, ctx, tbl_pref)
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_path}"}
            raw_result = _invoke_orchestrator(file_path, user_context, table_name_preference)
            if not isinstance(raw_result, str):  # safeguard: orchestrator should return JSON string
                raw_json_text = json.dumps(raw_result)
            else:
                raw_json_text = raw_result
            try:
                workflow_data = json.loads(raw_json_text)
            except Exception as parse_err:
                return {"success": False, "error": f"Failed to parse orchestrator JSON: {parse_err}", "raw": raw_json_text}
            return {
                "success": True,
                "workflow_status": workflow_data.get("workflow_status", "unknown"),
                "steps_completed": workflow_data.get("steps_completed", []),
                "file_analysis": workflow_data.get("step1_file_analysis", "Not available"),
                "rag_results": workflow_data.get("step2_rag_results", {}),
                "table_selection": workflow_data.get("step3_table_selection", {}),
                "generated_sql": workflow_data.get("step4_generated_sql", "Not available"),
                "ready_for_execution": workflow_data.get("ready_for_execution", False),
                "errors": workflow_data.get("errors", []),
                "_raw_json": workflow_data
            }
        except Exception as e:
            return {"success": False, "error": f"Unexpected orchestrator failure: {e}"}

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
                results.append(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
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
            
            # Separate statements if needed (useful to create a FIFO queue to process them automatically with an agent)
            separated = SQLCodeParser.separate_statements(clean_sql)
            
            return f"{clean_sql}"
        except Exception as e:
            return f"Error: Error generating SQL: {str(e)}"


processor = None

def initialize_processor():
    """Initialize the enhanced workflow processor"""
    global processor
    if processor is None:
        try:
            processor = WorkflowFileProcessor()
            return "Enhanced 4-Step Workflow Processor initialized successfully!"
        except Exception as e:
            return f"Error initializing processor: {str(e)}"
    return "Processor already initialized!"

def process_file_workflow(file, user_context: str = None, table_name: str = None) -> Tuple[str, str, str, str]:
    """Process uploaded file using the enhanced 4-step workflow."""
    if file is None:
        return "No file uploaded.", "", "", ""
    
    try:
        if hasattr(file, 'name'):
            file_path = file.name
        else:
            file_path = str(file)
        
        if not os.path.exists(file_path):
            return f"File not found: {file_path}", "", "", ""
        
        if processor is None:
            init_result = initialize_processor()
            if "Error" in init_result:
                return init_result, "", "", ""
        
        # Process with enhanced workflow
        result = processor.process_file_with_workflow(
            file_path, 
            user_context or f"File processing for database ingestion: {os.path.basename(file_path)}", 
            table_name
        )
        
        if not result["success"]:
            return f"Workflow failed: {result.get('error', 'Unknown error')}", "", "", ""
        
        # Format results for display
        workflow_summary = f"""WORKFLOW STATUS: {result['workflow_status'].upper()}
STEPS COMPLETED: {', '.join(result['steps_completed'])}
READY FOR EXECUTION: {'YES' if result.get('ready_for_execution') else 'NO'}
ERRORS: {'; '.join(result.get('errors', [])) if result.get('errors') else 'None'}

FILE: {os.path.basename(file_path)}"""

        # File analysis result (truncated for display)
        file_analysis = result.get('file_analysis', 'Not available')
        if len(file_analysis) > 2000:
            file_analysis = file_analysis[:2000] + "\n\n[... truncated for display ...]"
        
        # Table selection summary
        table_selection = result.get('table_selection', {})
        if isinstance(table_selection, dict) and 'selected_table' in table_selection:
            table_summary = f"""SELECTED TABLE: {table_selection['selected_table'].get('table_name', 'N/A')}
CONFIDENCE SCORE: {table_selection['selected_table'].get('confidence_score', 'N/A')}
SELECTION REASON: {table_selection['selected_table'].get('selection_reason', 'N/A')}

FIELD MAPPINGS: {len(table_selection.get('field_mappings', []))} mappings
UNMAPPED FIELDS: {len(table_selection.get('unmapped_fields', []))} fields
ESTIMATED SUCCESS RATE: {table_selection.get('estimated_success_rate', 'N/A')}"""
        else:
            table_summary = str(table_selection) if table_selection else "Table selection not available"
        
        # Generated SQL
        generated_sql = result.get('generated_sql', 'SQL not available')
        
        return workflow_summary, file_analysis, table_summary, generated_sql
        
    except Exception as e:
        return f"Error processing file: {str(e)}", "", "", ""

def process_question(question: str) -> str:
    """Process a general question using the enhanced agent."""
    if not question or not question.strip():
        return "Please provide a question."
    
    try:
        agent = EnhancedAgent()
        response = agent(question.strip())
        return response.strip()
        
    except Exception as e:
        return f"Error: {e}"

### GRADIO INTERFACE CREATION ###
def create_enhanced_gradio_interface():
    """Create an enhanced Gradio interface with 4-step workflow."""
    
    css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .workflow-panel {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        padding: 15px !important;
    }
    .output-text {
        font-family: 'Monaco', 'Consolas', monospace !important;
        font-size: 11px !important;
    }
    """
    
    with gr.Blocks(css=css, title="Enhanced File Processing with 4-Step Workflow") as interface:
        
        gr.Markdown("""
        # Enhanced Rules Engine Agent - 4-Step Database Ingestion Workflow
        ### Intelligent File Analysis → RAG Table Matching → Smart Selection → SQL Generation
        
        This enhanced system processes files through a sophisticated 4-step workflow for optimal database ingestion.
        """)
        
        # Initialization
        with gr.Row():
            init_btn = gr.Button("Initialize Enhanced Processor", variant="primary")
            init_status = gr.Textbox(label="Initialization Status", interactive=False)
            
        init_btn.click(fn=initialize_processor, outputs=init_status)
        
        gr.Markdown("---")
        
        # Enhanced file processing with workflow
        with gr.Tab("4-Step Workflow Processing"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Configuration")
                    
                    file_input = gr.File(
                        label="Upload File for Database Ingestion",
                        file_types=[".csv", ".xlsx", ".xls", ".json", ".txt", ".xml", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp"]
                    )
                    
                    user_context_input = gr.Textbox(
                        label="User Context (Optional)",
                        placeholder="e.g., 'Financial transaction data for quarterly reporting'",
                        lines=2
                    )
                    
                    table_name_input = gr.Textbox(
                        label="Preferred Table Name (Optional)",
                        placeholder="e.g., 'financial_transactions'",
                        value=""
                    )
                    
                    workflow_btn = gr.Button("Execute 4-Step Workflow", variant="primary", size="lg")
                    
                with gr.Column(scale=2):
                    gr.Markdown("### Workflow Results")
                    
                    workflow_status = gr.Textbox(
                        label="Workflow Summary",
                        lines=8,
                        interactive=False,
                        elem_classes=["workflow-panel"]
                    )
            
            # Detailed results in separate row
            with gr.Row():
                with gr.Column():
                    step1_output = gr.Textbox(
                        label="Step 1: File Analysis",
                        lines=12,
                        interactive=False,
                        elem_classes=["output-text"]
                    )
                    
                with gr.Column():
                    step3_output = gr.Textbox(
                        label="Step 3: Table Selection & Field Mapping",
                        lines=12,
                        interactive=False,
                        elem_classes=["output-text"]
                    )
            
            with gr.Row():
                step4_output = gr.Textbox(
                    label="Step 4: Generated SQL (Ready for Execution)",
                    lines=15,
                    interactive=False,
                    elem_classes=["output-text"]
                )
        
            workflow_btn.click(
                fn=process_file_workflow,
                inputs=[file_input, user_context_input, table_name_input],
                outputs=[workflow_status, step1_output, step3_output, step4_output]
            )
        
        # General agent chat
        with gr.Tab("General Agent Chat"):
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Ask the Enhanced Agent",
                        placeholder="Try: 'Please ingest the cobalt_global_maj-rej-0001_2706.csv file into our database'",
                        lines=3
                    )
                    
                    chat_btn = gr.Button("Ask Agent", variant="primary")
                    
                with gr.Column():
                    chat_output = gr.Textbox(
                        label="Agent Response",
                        lines=20,
                        interactive=False
                    )
            
            chat_btn.click(
                fn=process_question,
                inputs=question_input,
                outputs=chat_output
            )
    
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
            interface = create_enhanced_gradio_interface()
            interface.launch(
                server_name="127.0.0.1",
                server_port=7860,
                share=True,
                show_error=False,
                debug=False
            )
        except Exception as e:
            print(f"Error launching Gradio interface: {e}")
            print("Trying fallback configuration...")
            
            # Fallback
            try:
                interface = create_enhanced_gradio_interface()
                interface.launch(share=True, debug=False)
            except Exception as e2:
                print(f"Failed to launch interface: {e2}")
                print("Please try running with --cli flag for command line mode")
                print("Example: python app.py --cli")

if __name__ == "__main__":
    main()

# === Showcase Enhancements (frontend only, no changes to agent/tools/RAG) ===
last_workflow_artifacts = {  # stores latest successful workflow outputs for download
    "sql": None,
    "table_selection": None,
    "raw_json": None,
    "file_name": None
}

# Helper to create temp file for downloads
import tempfile

def _write_temp_file(content: str, suffix: str) -> str:
    try:
        if not content:
            return ""
        fd, path = tempfile.mkstemp(suffix=suffix, text=True)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        return path
    except Exception:
        return ""

def download_sql_file() -> str:
    if last_workflow_artifacts.get("sql"):
        return _write_temp_file(last_workflow_artifacts["sql"], ".sql")
    return ""

def download_mapping_file() -> str:
    mapping = last_workflow_artifacts.get("table_selection")
    if isinstance(mapping, dict):
        return _write_temp_file(json.dumps(mapping, indent=2), ".json")
    return ""

def download_full_json_file() -> str:
    raw = last_workflow_artifacts.get("raw_json")
    if raw:
        return _write_temp_file(json.dumps(raw, indent=2), ".json")
    return ""

# Patch process_file_workflow to persist artifacts (frontend only)
_original_process_file_workflow = process_file_workflow

def process_file_workflow(file, user_context: str = None, table_name: str = None):  # noqa: F811
    summary, analysis, mapping_summary, sql = _original_process_file_workflow(file, user_context, table_name)
    # Attempt to capture raw artifacts if workflow succeeded
    try:
        if summary.startswith("WORKFLOW STATUS") and file:
            # Re-run minimal orchestrator call to capture full JSON (already done inside processor; reuse result)
            file_path = file.name if hasattr(file, 'name') else str(file)
            orch_raw = database_ingestion_orchestrator.func(file_path=file_path, user_context=user_context, table_name_preference=table_name) if hasattr(database_ingestion_orchestrator, 'func') else database_ingestion_orchestrator(file_path, user_context, table_name)
            parsed = json.loads(orch_raw) if isinstance(orch_raw, str) else orch_raw
            last_workflow_artifacts["raw_json"] = parsed
            last_workflow_artifacts["sql"] = parsed.get("step4_generated_sql")
            last_workflow_artifacts["table_selection"] = parsed.get("step3_table_selection")
            last_workflow_artifacts["file_name"] = os.path.basename(file_path)
    except Exception:
        pass
    return summary, analysis, mapping_summary, sql

# Extend UI creation
_create_enhanced_gradio_interface_original = create_enhanced_gradio_interface

def create_enhanced_gradio_interface():  # noqa: F811
    interface = gr.Blocks(css="""
    .gradio-container {max-width: 1500px !important;}
    .artifact-box {font-family: 'Consolas', monospace; font-size: 11px;}
    """, title="4-Step Ingestion Showcase")

    with interface:
        gr.Markdown("""
        # 4-Step Agentic Data Ingestion - Proof of Concept Showcase
        This interface demonstrates an automated pipeline turning a raw file into executable SQL via:
        1. File Analysis  
        2. RAG Table Retrieval  
        3. Intelligent Table Selection & Field Mapping  
        4. SQL Generation  
        """)

        with gr.Tab("Showcase Overview"):
            gr.Markdown("""
            ## Architecture Overview
            **Goal:** Automate ingestion readiness from arbitrary file to database SQL.
            **Flow:** File → Analysis → RAG Candidates → Selection & Mapping → SQL.
            **Outputs:**
            - Structured file analysis
            - Top candidate tables with semantic scores
            - Selected target table + field mappings
            - Generated CREATE TABLE + sample INSERT statements

            ### Components
            - Orchestrator Tool: Chains the four steps deterministically
            - RAG Layer: Qdrant vector store (schema embeddings)
            - LLM Reasoning: Mapping + SQL synthesis
            - UI: This Gradio app for presentation & inspection

            ### Demo Tips
            1. Upload or reuse a sample CSV (e.g., cobalt_global_maj-rej-0001_2706.csv)
            2. Optionally add context ("transaction outcomes") or a preferred table name
            3. Run the workflow and inspect outputs across panels
            4. Download artifacts (SQL, mappings, full JSON)
            """)

        with gr.Tab("Ingestion Workflow"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload File", file_types=[".csv", ".xlsx", ".xls", ".json", ".txt", ".xml", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp"])
                    user_context_input = gr.Textbox(label="User Context (Optional)", placeholder="e.g., transaction outcomes for analytics", lines=2)
                    table_name_input = gr.Textbox(label="Preferred Table Name (Optional)", placeholder="e.g., operations_staging")
                    run_btn = gr.Button("Run 4-Step Workflow", variant="primary")
                    with gr.Accordion("Download Artifacts", open=False):
                        sql_download = gr.File(label="Download Generated SQL", interactive=False)
                        mapping_download = gr.File(label="Download Field Mapping JSON", interactive=False)
                        full_json_download = gr.File(label="Download Full Workflow JSON", interactive=False)
                        refresh_btn = gr.Button("Refresh Downloads")
                with gr.Column(scale=2):
                    summary_box = gr.Textbox(label="Workflow Summary", lines=8, interactive=False, elem_classes=["artifact-box"])
                    with gr.Row():
                        analysis_box = gr.Textbox(label="Step 1: File Analysis", lines=14, interactive=False, elem_classes=["artifact-box"])
                        mapping_box = gr.Textbox(label="Step 3: Table Selection & Mapping", lines=14, interactive=False, elem_classes=["artifact-box"])
                    sql_box = gr.Textbox(label="Step 4: Generated SQL", lines=16, interactive=False, elem_classes=["artifact-box"])
                    with gr.Accordion("Raw JSON Result (Full)", open=False):
                        raw_json_display = gr.JSON(label="Complete Orchestrator Output")

            def _refresh_json():
                return last_workflow_artifacts.get("raw_json") or {}

            def _refresh_downloads():
                return (
                    download_sql_file(),
                    download_mapping_file(),
                    download_full_json_file()
                )

            def _workflow_trigger(file, ctx, tbl):
                summary, analysis, mapping, sql = process_file_workflow(file, ctx, tbl)
                raw = last_workflow_artifacts.get("raw_json") or {}
                return summary, analysis, mapping, sql, raw

            run_btn.click(
                fn=_workflow_trigger,
                inputs=[file_input, user_context_input, table_name_input],
                outputs=[summary_box, analysis_box, mapping_box, sql_box, raw_json_display]
            )

            refresh_btn.click(fn=_refresh_downloads, outputs=[sql_download, mapping_download, full_json_download])
            # Auto-refresh downloads after run
            run_btn.click(fn=_refresh_downloads, outputs=[sql_download, mapping_download, full_json_download])

        with gr.Tab("Agent Chat"):
            chat_query = gr.Textbox(label="Ask the Agent", placeholder="e.g., Ingest cobalt_global_maj-rej-0001_2706.csv with table name operations_demo", lines=3)
            chat_btn = gr.Button("Send", variant="primary")
            chat_answer = gr.Textbox(label="Agent Response", lines=15, interactive=False)

            def _agent_chat(q):
                return process_question(q)
            chat_btn.click(fn=_agent_chat, inputs=chat_query, outputs=chat_answer)

    return interface

# Replace original creator reference
# ...existing code...