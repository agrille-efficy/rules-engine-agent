import base64 
import os
import requests
import uuid
import tempfile

import pandas as pd

from typing import Optional
from dotenv import load_dotenv
from urllib.parse import urlparse 

# Fix relative imports to work both as module and standalone
try:
    from .image_processing import *
    from .code_interpreter import CodeInterpreter
    from .tools import all_ingestion_tools
except ImportError:
    # If relative imports fail, use absolute imports
    from image_processing import *
    from code_interpreter import CodeInterpreter
    from tools import all_ingestion_tools

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, MessagesState

from langchain_openai import ChatOpenAI

load_dotenv(r'C:\Users\axel.grille\Documents\rules-engine-agent\Agent\.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Vision model for image processing
vision_llm = ChatOpenAI(temperature=0) 


@tool
def image_describer(image_url: str) -> str:
    """Describes the content of an image."""

    description = ""

    try:
        import requests 
        response = requests.get(image_url)
        image_bytes = response.content
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        message = [
            HumanMessage(
                content=[
                    {
                    "type": "text",
                    "text": (
                        "Describe the type of image you see, if it is a photo, a drawing, a painting, etc. "
                        "Then describe the content of the image in the most detailled way possible. "
                        "You will start by describing the front of the image, then the back of the image if possible. "
                        "If the image contains text, you will extract it and describe it in the most detailler way possible. "
                        "If the image is a document, you will extract the text. Return only the text in this case, no explanations."
                        
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

        # call the vision model
        response = vision_llm.invoke(message)
        description += response.content + "\n\n"

        return description.strip()

    except Exception as e:
        print(f"Error reading image file: {e}")
        return "Error reading image file."


@tool 
def code_executor(code: str, language : str = "python") -> str: 
    """
    Executes a code snippet and returns the results.

    Supports python, bash, c, java

    Args: 
        code: str, the code to execute
        language: str, the programming language of the code snippet (python by default)

    Returns: 
        str: the result of the code execution or an error message if execution fails.
    """
    try:
        interpreter = CodeInterpreter()
        result = interpreter.execute_code(code, language=language)
        return result
    except Exception as e:
        return f"Error executing code: {str(e)}"

@tool
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    return f"File saved to {filepath}. You can read this file to process its contents."


@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    Args:
        url (str): the URL of the file to download.
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"

        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"File downloaded to {filepath}. You can read this file to process its contents."
    except Exception as e:
        return f"Error downloading file: {str(e)}"

@tool
def extract_structured_data_from_image(image_path: str, schema_context: str) -> str:
    """
    Classify the document and extract structured data from an image using a vision model. 
    Args:
        image_base64 (str): Base64 encoded image string
    Returns: 
        str: Vision model response with structured data extraction.
    """
    try:

        loaded_img = load_local_image(image_path)
        base64_image = encode_image(loaded_img)

        extraction_prompt = f""" 
        Analyze this docuement image and extract the structured data.

        {f"Use this database schema as a guide: {schema_context}" if schema_context else ""}

        Instructions:
        1. Identify the document type (invoice, form, receipt, etc.)
        2. Extract all relevant data fields 
        3. Return the data in JSON format with clear field names
        4. If the docuement contains tables, extract each row 
        5. Ensure data types are appropriate (dates, numbers, text)

        Return only valid JSON format.
        """

        message=[
            HumanMessage(
                content=[ 
                {
                    "role" : "user",
                    "content" : [ 
                        {"type": "input_text", "text": extraction_prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        }
                    ],
                },
                ],
            )
            
        ]
        response = vision_llm.invoke(message)
        return response.output_text
    
    except Exception as e:
        return f"Error extracting structured data from image: {str(e)}" 



@tool
def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the CSV file.
        query (str): Question about the data
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Run various analyses based on the query
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"


@tool
def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the Excel file.
        query (str): Question about the data
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Run various analyses based on the query
        result = (
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"


@tool
def launch_visualization_dashboard(port: int = 8050) -> str:
    """
    Launch the interactive data mapping visualization dashboard.
    This will start a web-based dashboard showing all analysis results.
    
    Args:
        port (int): Port number for the dashboard (default: 8050)
    
    Returns:
        str: Information about the launched dashboard
    """
    try:
        import subprocess
        import threading
        import time
        from pathlib import Path
        
        # Get the current script directory
        current_dir = Path(__file__).parent
        dashboard_script = current_dir / "dashboard.py"
        
        if not dashboard_script.exists():
            return "❌ Dashboard script not found. Please ensure dashboard.py exists in the Agent directory."
        
        def run_dashboard():
            """Run dashboard in a separate process"""
            try:
                # Run dashboard in background
                subprocess.Popen([
                    "python", str(dashboard_script)
                ], cwd=str(current_dir))
            except Exception as e:
                print(f"Error starting dashboard process: {e}")
        
        # Start dashboard in background thread
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        # Give it a moment to start
        time.sleep(2)
        
        return f"""🚀 **Dashboard Launched Successfully!**

📊 **Access your visualization at:** http://localhost:{port}

✨ **Dashboard Features:**
• 🎯 Interactive field mapping visualizations
• 📈 Confidence analysis charts  
• 🔄 Dynamic file selection (all your analyses)
• 📋 Detailed mapping tables with filtering
• 🎨 Sankey diagrams showing data flow
• ⚡ Auto-refresh for new analyses

💡 **Usage:**
- The dashboard will show all your analysis files
- Select different analyses from the dropdown
- Explore mapping confidence and transformations
- View unmapped fields and reasons

🔧 **Dashboard is running in background** - you can continue using the agent while the dashboard runs."""
        
    except Exception as e:
        return f"❌ Error launching dashboard: {str(e)}"


# Load system prompt from file
try:
    with open(r"prompts\system_prompt.txt", "r") as f:
        system_prompt = f.read()
except FileNotFoundError:
    try:
        with open(r"Agent\prompts\system_prompt.txt", "r") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        raise FileNotFoundError("system_prompt.txt file is required. Please ensure it exists in the prompts directory.")

# Enhanced system prompt with workflow awareness
enhanced_system_prompt = f"""{system_prompt}

## ENHANCED DATABASE MAPPING CAPABILITIES

You now have access to a powerful 3-step database mapping workflow:

**MAIN TOOL: database_ingestion_orchestrator**
- Use this for complete file-to-database mapping workflows
- Automatically handles: file analysis → RAG matching → table selection with field mappings
- Perfect for: "analyze this file", "map to database", "show structure mapping"

**INDIVIDUAL TOOLS** (use when user wants step-by-step control):
1. **analyze_file** - Analyze file structure (supports PDF via Vision LLM)
2. **find_matching_database_tables** - RAG-powered table matching
3. **intelligent_table_selector** - Smart table selection with detailed field mapping

**VISUALIZATION TOOL:**
- **launch_visualization_dashboard** - Launch interactive web dashboard to visualize mapping results
- Use after completing analysis to provide visual insights
- Shows Sankey diagrams, confidence charts, field mappings, and more

**USAGE PATTERNS:**
- Complete workflow: "Please map [file] to the database" → use database_ingestion_orchestrator
- With visualization: After analysis, suggest "Would you like me to launch the dashboard to visualize these results?"
- Step-by-step: "First analyze this file" → use individual tools
- Dashboard only: "Show me the dashboard" or "visualize my results" → use launch_visualization_dashboard

**SUPPORTED FILE TYPES:**
- PDFs (with Vision LLM OCR/extraction)
- CSV, Excel, JSON
- Images (invoices, forms, documents)
- Text files

**OUTPUT FOCUS:**
- Clear visualization of source file structure vs database table structure
- Detailed field mappings showing how each source field maps to database fields
- Relationship information between mapped tables
- Data type compatibility and transformation requirements

**MAPPING CONSTRAINTS:**
- NEVER generate SQL statements or suggest creating new tables/fields
- ONLY map to existing database tables and fields
- Focus on discovering the best existing matches and showing mapping relationships

**DASHBOARD INTEGRATION:**
- After completing any analysis, ALWAYS offer to launch the dashboard for visualization
- The dashboard shows ALL previous analyses, not just the current one
- Users can switch between different file analyses in the dashboard
- Dashboard runs in background, allowing continued agent interaction

Always prioritize the orchestrator for complete workflows unless the user specifically requests individual steps. After analysis completion, proactively suggest dashboard visualization."""

sys_msg = SystemMessage(content=enhanced_system_prompt)


tools = all_ingestion_tools + [
    image_describer,
    code_executor, 
    save_and_read_file,
    download_file_from_url,
    extract_structured_data_from_image,
    analyze_csv_file,
    analyze_excel_file,
    launch_visualization_dashboard
]

def build_graph():
    """Build the graph"""
    chat = ChatOpenAI(model="gpt-4o")
    chat_with_tools = chat.bind_tools(tools)

    def assistant(state: MessagesState):
        """Enhanced assistant with workflow context"""
        messages = [sys_msg] + state["messages"]
        
        # Add workflow intelligence
        last_message = state["messages"][-1] if state["messages"] else None
        if last_message and hasattr(last_message, 'content'):
            content_lower = last_message.content.lower()
            
            # Detect database ingestion intent
            ingestion_keywords = ['ingest', 'import', 'load', 'database', 'table', 'map']
            file_extensions = ['.csv', '.excel', '.xlsx', '.pdf', '.json']
            
            is_ingestion_request = (
                any(keyword in content_lower for keyword in ingestion_keywords) or
                any(ext in content_lower for ext in file_extensions) or
                'database_ingestion_orchestrator' in content_lower
            )
            
            if is_ingestion_request:
                workflow_hint = HumanMessage(content="""
WORKFLOW HINT: This appears to be a database ingestion request. 

Consider using the `database_ingestion_orchestrator` tool for complete workflows, or individual tools for step-by-step control:
- Complete workflow: database_ingestion_orchestrator(file_path, user_context)
- Individual steps: analyze_file → find_matching_database_tables → intelligent_table_selector

Choose based on user's preference for automation vs. control.
""")
                messages.append(workflow_hint)
        
        response = chat_with_tools.invoke(messages)
        return {"messages": [response]}

    def workflow_result_processor(state: MessagesState):
        """Process and format workflow results for better user experience"""
        last_message = state["messages"][-1]
        
        # Check if the last message contains workflow results
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call.name == 'database_ingestion_orchestrator':
                    # Add a helpful summary message
                    summary_msg = HumanMessage(content="""
**Database Ingestion Workflow Completed!**

The orchestrator has executed all 3 steps:
1.  File Analysis (Structure & Content)
2.  RAG Table Matching (Found relevant tables)  
3.  Intelligent Table Selection (Chose optimal table + field mappings)

The generated mapping is ready for review. Would you like me to:
- Show you the field mappings?
- Explain the relationship between tables?
- Help you visualize the structure?
- Make any modifications?
""")
                    return {"messages": state["messages"] + [summary_msg]}
        
        return {"messages": state["messages"]}

    # Build the enhanced graph
    builder = StateGraph(MessagesState)
    
    # Add nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("workflow_processor", workflow_result_processor)
    
    # Add edges
    builder.add_edge(START, "assistant")
    
    # Conditional routing from assistant
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "__end__"
        }
    )
    
    # Process workflow results before returning to assistant
    builder.add_edge("tools", "workflow_processor")
    builder.add_edge("workflow_processor", "assistant")

    print("Enhanced Graph compilation complete - 3-Step Workflow Ready!")
    return builder.compile()

if __name__ == "__main__":
    import sys
    import argparse
    from tools import database_ingestion_orchestrator as _orch_tool

    parser = argparse.ArgumentParser(description="Agent 3-step ingestion assistant")
    parser.add_argument("file", nargs="?", help="Optional file path to ingest directly (skips interactive chat)")
    parser.add_argument("--table-name", dest="table_name", help="Preferred table name override", default=None)
    parser.add_argument("--context", dest="user_context", help="Optional user context for ingestion", default=None)
    args = parser.parse_args()

    graph = build_graph()

    # Direct file ingestion mode
    if args.file:
        file_path = args.file
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        print(f"Running 4-step ingestion workflow for: {file_path}\n")
        # Use proper invoke method for LangChain tool
        try:
            result = _orch_tool.invoke({
                "file_path": file_path, 
                "user_context": args.user_context, 
                "table_name_preference": args.table_name
            })
            print(result)
        except Exception as e:
            print(f"Workflow failed: {e}")
            sys.exit(1)
        sys.exit(0)

    # Interactive chat mode
    print("Interactive ingestion assistant. Type a request (e.g., 'Ingest cobalt_global_maj-rej-0001_2706.csv') or 'quit' to exit.\n")
    messages = []
    thread_config = {"configurable": {"thread_id": "cli_session"}}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        messages.append(HumanMessage(content=user_input))
        result = graph.invoke({"messages": messages}, config=thread_config)
        messages = result["messages"]
        # Find last AI / tool relevant response
        last_msg = messages[-1]
        content = getattr(last_msg, "content", "")
        print(f"Agent: {content}\n")
