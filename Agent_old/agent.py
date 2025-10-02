import base64
import os
import requests
import uuid
import tempfile
import json
import argparse
from typing import Optional, Literal

import pandas as pd

from dotenv import load_dotenv
from urllib.parse import urlparse

from image_processing import encode_image, load_local_image
from code_interpreter import CodeInterpreter

from langchain_core.tools import tool

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI

load_dotenv(r'C:\Users\axel.grille\Documents\rules-engine-agent\Agent\.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)


# class WorkflowState(TypedDict, total=False):
#     messages: list
#     file_path: str
#     user_context: Optional[str]
#     table_preference: Optional[str]
#     user_preferences: Optional[str]

#     file_analysis_result: Optional[str]
#     rag_results: Optional[dict]
#     selected_table: Optional[dict]
#     validation_result: Optional[dict]

#     workflow_step: str
#     validation_status: Optional[str]
#     refinement_attempts: int
#     max_refinements: int
#     workflow_status: str
#     errors: list
#     last_error: Optional[str]
#     steps_completed: list
#     refinement_history: list


@tool
def image_describer(image_url: str) -> str:
    """Describes the content of an image."""
    description = ""
    try:
        response = requests.get(image_url)
        image_bytes = response.content
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Describe the type of image you see, "
                            "if it is a photo, a drawing, a painting, etc. "
                            "Then describe the content of the image "
                            "in the most detailed way possible. "
                            "You will start by describing the front of the "
                            "image, then the back of the image if possible. "
                            "If the image contains text, you will extract it "
                            "and describe it in the most detailed way possible. "
                            "If the image is a document, you will extract the text. "
                            "Return only the text in this case, no explanations."
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
        description += response.content + "\n\n"
        return description.strip()
    except Exception as e:
        print(f"Error reading image file: {e}")
        return "Error reading image file."


@tool
def code_executor(code: str, language: str = "python") -> str:
    """
    Executes a code snippet and returns the results.
    Supports python, bash, c, java
    Args:
        code: str, the code to execute
        language: str, the programming language of the code snippet
            (python by default)
    Returns:
        str: the result of the code execution or an error message if
        execution fails.
    """
    try:
        interpreter = CodeInterpreter()
        result = interpreter.execute_code(code, language=language)
        return result
    except Exception as e:
        return f"Error executing code: {str(e)}"


@tool
def save_and_read_file(
    content: str, filename: Optional[str] = None
) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file.
        If not provided, a random name file will be created.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)
    with open(filepath, "w") as f:
        f.write(content)
    return (
        f"File saved to {filepath}. You can read this file to process its "
        f"contents."
    )


@tool
def download_file_from_url(
    url: str, filename: Optional[str] = None
) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    Args:
        url (str): the URL of the file to download.
        filename (str, optional): the name of the file. If not provided,
            a random name file will be created.
    """
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return (
            f"File downloaded to {filepath}. You can read this file to "
            f"process its contents."
        )
    except Exception as e:
        return f"Error downloading file: {str(e)}"


@tool
def extract_structured_data_from_image(
    image_path: str, schema_context: str
) -> str:
    """
    Classify the document and extract structured data from an image using
    a vision model.
    Args:
        image_base64 (str): Base64 encoded image string
    Returns:
        str: Vision model response with structured data extraction.
    """
    try:
        loaded_img = load_local_image(image_path)
        base64_image = encode_image(str(loaded_img))
        schema_guide = (
            f"Use this database schema as a guide: {schema_context}"
            if schema_context else ""
        )
        extraction_prompt = f"""
        Analyze this document image and extract the structured data.
        {schema_guide}
        Instructions:
        1. Identify the document type (invoice, form, receipt, etc.)
        2. Extract all relevant data fields
        3. Return the data in JSON format with clear field names
        4. If the document contains tables, extract each row
        5. Ensure data types are appropriate (dates, numbers, text)
        Return only valid JSON format.
        """
        message = [
            HumanMessage(
                content=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": extraction_prompt},
                            {
                                "type": "input_image",
                                "image_url": (
                                    f"data:image/png;base64,{base64_image}"
                                ),
                            }
                        ],
                    },
                ],
            )
        ]
        response = vision_llm.invoke(message)
        return str(response.content)
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
        df = pd.read_csv(file_path)
        result = (
            f"CSV file loaded with {len(df)} rows and "
            f"{len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"
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
        df = pd.read_excel(file_path)
        result = (
            f"Excel file loaded with {len(df)} rows and "
            f"{len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"
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
        current_dir = Path(__file__).parent
        dashboard_script = current_dir / "dashboard.py"
        if not dashboard_script.exists():
            return (
                "ERROR: Dashboard script not found. Please ensure "
                "dashboard.py exists in the Agent directory."
            )

        def run_dashboard():
            try:
                subprocess.Popen([
                    "python", str(dashboard_script)
                ], cwd=str(current_dir))
            except Exception as e:
                print(f"Error starting dashboard process: {e}")
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        time.sleep(2)
        return f"""DASHBOARD LAUNCHED SUCCESSFULLY!
            Access your visualization at: http://localhost:{port}
            Dashboard Features:
            • Interactive field mapping visualizations
            • Confidence analysis charts
            • Dynamic file selection (all your analyses)
            • Detailed mapping tables with filtering
            • Sankey diagrams showing data flow
            • Auto-refresh for new analyses
            Usage:
            - The dashboard will show all your analysis files
            - Select different analyses from the dropdown
            - Explore mapping confidence and transformations
            - View unmapped fields and reasons
            Dashboard is running in background - you can continue using the
            agent while the dashboard runs."""
    except Exception as e:
        return f"ERROR: Error launching dashboard: {str(e)}"


# ============================================================================

def file_analysis_node(state: WorkflowState) -> WorkflowState:
    """
    STEP 1: Analyze file structure and content
    Calls the analyze_file tool from tools.py
    """
    file_path = state['file_path']
    print(f"STEP 1: Analyzing file structure and content for "
          f"{file_path}...")

    from tools import analyze_file

    try:
        result = analyze_file.invoke({"file_path": state["file_path"]})

        if result.startswith("Error"):
            return {
                **state,
                "last_error": result,
                "errors": (
                    state.get("errors", []) +
                    [f"File Analysis: {result}"]
                ),
                "workflow_step": "error",
                "workflow_status": "failed"
            }

        print(f"  File analyzed successfully - {len(result)} characters")

        basename = os.path.basename(state['file_path'])
        return {
            **state,
            "file_analysis_result": result,
            "workflow_step": "rag_matching",
            "steps_completed": (
                state.get("steps_completed", []) + ["file_analysis"]
            ),
            "messages": state.get("messages", []) + [
                HumanMessage(
                    content=f"File analysis completed for {basename}"
                )
            ]
        }

    except Exception as e:
        error_msg = f"Exception in file analysis: {str(e)}"
        print(f"  ERROR: {error_msg}")
        return {
            **state,
            "last_error": error_msg,
            "errors": state.get("errors", []) + [error_msg],
            "workflow_step": "error",
            "workflow_status": "failed"
        }


def rag_matching_node(state: WorkflowState) -> WorkflowState:
    """
    STEP 2: Find matching database tables using RAG
    Calls the find_matching_database_tables tool from tools.py
    """
    print("STEP 2: Finding matching database tables using RAG...")

    from tools import find_matching_database_tables

    try:
        result = find_matching_database_tables.invoke({
            "file_path": state["file_path"],
            "user_context": state.get("user_context", "")
        })

        if result.startswith("Error"):
            return {
                **state,
                "last_error": result,
                "errors": (
                    state.get("errors", []) + [f"RAG Matching: {result}"]
                ),
                "workflow_step": "error",
                "workflow_status": "failed"
            }

        # Parse JSON result
        try:
            rag_data = json.loads(result)
            top_tables_count = len(rag_data.get("top_10_tables", []))
            print(f"  Found {top_tables_count} matching tables")

            return {
                **state,
                "rag_results": rag_data,
                "workflow_step": "table_selection",
                "steps_completed": (
                    state.get("steps_completed", []) + ["rag_matching"]
                ),
                "messages": state.get("messages", []) + [
                    HumanMessage(
                        content=(
                            f"RAG matching completed - found "
                            f"{top_tables_count} matching tables"
                        )
                    )
                ]
            }

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse RAG results: {str(e)}"
            print(f"  ERROR: {error_msg}")
            return {
                **state,
                "last_error": error_msg,
                "errors": state.get("errors", []) + [error_msg],
                "workflow_step": "error",
                "workflow_status": "failed"
            }

    except Exception as e:
        error_msg = f"Exception in RAG matching: {str(e)}"
        print(f"  ERROR: {error_msg}")
        return {
            **state,
            "last_error": error_msg,
            "errors": state.get("errors", []) + [error_msg],
            "workflow_step": "error",
            "workflow_status": "failed"
        }


def table_selection_node(state: WorkflowState) -> WorkflowState:
    """
    STEP 3: Select optimal table and create field mappings
    Calls the intelligent_table_selector tool from tools.py
    """
    attempt_num = state.get("refinement_attempts", 0) + 1
    print(f"STEP 3 (Attempt {attempt_num}): Selecting optimal table and "
          f"creating field mappings...")

    from tools import intelligent_table_selector

    try:
        # Build user preferences including refinement suggestions
        user_prefs = []
        if state.get("table_preference"):
            user_prefs.append(
                f"Table preference: {state['table_preference']}"
            )

        # Add refinement suggestions from previous validation
        validation_result = state.get("validation_result")
        if (validation_result and
                validation_result.get("refinement_suggestions")):
            suggestions = validation_result["refinement_suggestions"]
            print(f"  Applying {len(suggestions)} refinement suggestions "
                  f"from previous attempt")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"    {i}. {suggestion}")
            user_prefs.extend(suggestions)

        result = intelligent_table_selector.invoke({
            "file_analysis": state["file_analysis_result"],
            "rag_results": json.dumps(state["rag_results"]),
            "user_preferences": " | ".join(user_prefs) if user_prefs else ""
        })

        if result.startswith("Error"):
            return {
                **state,
                "last_error": result,
                "errors": (
                    state.get("errors", []) +
                    [f"Table Selection: {result}"]
                ),
                "workflow_step": "error",
                "workflow_status": "failed"
            }

        # Parse result
        try:
            selection_data = json.loads(result)
            selected_table = selection_data.get("selected_table", {})
            table_name = selected_table.get("table_name", "Unknown")
            confidence = selected_table.get("confidence_score", 0.0)
            mappings_count = len(selection_data.get("field_mappings", []))
            unmapped_count = len(selection_data.get("unmapped_fields", []))

            print(f"  Selected table: {table_name} "
                  f"(confidence: {confidence:.2f})")
            print(f"  Field mappings: {mappings_count} mapped, "
                  f"{unmapped_count} unmapped")

            return {
                **state,
                "selected_table": selection_data,
                "workflow_step": "validation",
                "steps_completed": (
                    state.get("steps_completed", []) + ["table_selection"]
                ),
                "messages": state.get("messages", []) + [
                    HumanMessage(
                        content=(
                            f"Table selected: {table_name} with "
                            f"{mappings_count} field mappings"
                        )
                    )
                ]
            }

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse table selection results: {str(e)}"
            print(f"  ERROR: {error_msg}")
            return {
                **state,
                "last_error": error_msg,
                "errors": state.get("errors", []) + [error_msg],
                "workflow_step": "error",
                "workflow_status": "failed"
            }

    except Exception as e:
        error_msg = f"Exception in table selection: {str(e)}"
        print(f"  ERROR: {error_msg}")
        return {
            **state,
            "last_error": error_msg,
            "errors": state.get("errors", []) + [error_msg],
            "workflow_step": "error",
            "workflow_status": "failed"
        }


def validation_node(state: WorkflowState) -> WorkflowState:
    """
    STEP 4: Validate mapping quality and determine next action
    Calls the results_analyzer tool from tools.py
    """
    attempt_num = state.get("refinement_attempts", 0) + 1
    print(f"STEP 4 (Attempt {attempt_num}): Validating mapping quality...")

    from tools import results_analyzer

    try:
        result = results_analyzer.invoke({
            "step3_results": json.dumps(state["selected_table"]),
            "rag_results": json.dumps(state["rag_results"]),
            "table_to_ingest": state.get("table_preference")
        })

        if result.startswith("Error"):
            return {
                **state,
                "last_error": result,
                "errors": (
                    state.get("errors", []) + [f"Validation: {result}"]
                ),
                "workflow_step": "error",
                "workflow_status": "failed"
            }

        # Parse validation result
        try:
            validation_data = json.loads(result)
            recommendation = validation_data.get("recommendation")
            confidence = validation_data.get("confidence", 0.0)
            reasoning = validation_data.get("reasoning", "")

            print("  VALIDATION RESULTS:")
            print(f"    Recommendation: {recommendation.upper()}")
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Reasoning: {reasoning[:100]}...")

            # Log quality metrics
            analysis = validation_data.get("analysis", {})
            mapping_quality = analysis.get("mapping_quality", {})
            print("  QUALITY METRICS:")
            conf_score = mapping_quality.get('confidence_score', 0.0)
            print(f"    Mapping confidence: {conf_score:.2f}")
            coverage = mapping_quality.get('field_coverage', 0.0)
            print(f"    Field coverage: {coverage:.1%}")

            # Record refinement history
            refinement_history = state.get("refinement_history", [])
            refinement_history.append({
                "attempt": attempt_num,
                "table_selection": state["selected_table"],
                "validation": validation_data,
                "recommendation": recommendation
            })

            return {
                **state,
                "validation_result": validation_data,
                "validation_status": recommendation,
                "refinement_history": refinement_history,
                "workflow_step": "decision",
                "steps_completed": (
                    state.get("steps_completed", []) + ["validation"]
                ),
                "messages": state.get("messages", []) + [
                    HumanMessage(
                        content=(
                            f"Validation completed: {recommendation} "
                            f"(confidence: {confidence:.2f})"
                        )
                    )
                ]
            }

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse validation results: {str(e)}"
            print(f"  ERROR: {error_msg}")
            return {
                **state,
                "last_error": error_msg,
                "errors": state.get("errors", []) + [error_msg],
                "workflow_step": "error",
                "workflow_status": "failed"
            }

    except Exception as e:
        error_msg = f"Exception in validation: {str(e)}"
        print(f"  ERROR: {error_msg}")
        return {
            **state,
            "last_error": error_msg,
            "errors": state.get("errors", []) + [error_msg],
            "workflow_step": "error",
            "workflow_status": "failed"
        }


def error_handler_node(state: WorkflowState) -> WorkflowState:
    """
    ERROR HANDLER: Handle workflow errors gracefully
    Provides error analysis and recovery suggestions
    """
    print("\n=== ERROR HANDLER ===")

    last_error = state.get("last_error", "Unknown error")
    errors = state.get("errors", [])
    current_step = state.get("workflow_step", "unknown")

    print(f"Error occurred at step: {current_step}")
    print(f"Last error: {last_error}")
    print(f"Total errors: {len(errors)}")

    # Analyze error and provide recovery suggestions
    recovery = _generate_recovery_suggestions(
        last_error or "Unknown error", current_step
    )
    error_analysis = {
        "error_type": _classify_error(last_error),
        "failed_step": current_step,
        "error_message": last_error,
        "all_errors": errors,
        "recovery_suggestions": recovery
    }

    # Create error report
    error_report = f"""
ERROR REPORT
============
Failed Step: {current_step}
Error Type: {error_analysis['error_type']}
Error Message: {last_error}

Recovery Suggestions:
"""
    suggestions = error_analysis['recovery_suggestions']
    if isinstance(suggestions, list):
        for i, suggestion in enumerate(suggestions, 1):
            error_report += f"{i}. {suggestion}\n"

    print(error_report)

    return {
        **state,
        "workflow_status": "failed",
        "workflow_step": "completed_with_errors",
        "error_analysis": error_analysis,
        "messages": state.get("messages", []) + [
            HumanMessage(
                content=f"Workflow failed at {current_step}: {last_error}"
            )
        ]
    }


def refinement_node(state: WorkflowState) -> WorkflowState:
    """
    REFINEMENT: Prepare for another refinement attempt
    Increments refinement counter and routes back to table selection
    """
    current_attempts = state.get("refinement_attempts", 0)
    max_attempts = state.get("max_refinements", 3)

    print("\n=== REFINEMENT NODE ===")
    print(f"Current attempt: {current_attempts}")
    print(f"Max attempts: {max_attempts}")

    # Check if we've reached max refinements
    if current_attempts >= max_attempts:
        print(f"Max refinement attempts ({max_attempts}) reached")
        print("Escalating to manual review")

        return {
            **state,
            "workflow_status": "max_refinements_reached",
            "workflow_step": "completed",
            "validation_status": "manual_review",
            "messages": state.get("messages", []) + [
                HumanMessage(
                    content=(
                        f"Max refinement attempts reached ({max_attempts}). "
                        f"Manual review required."
                    )
                )
            ]
        }

    # Increment refinement counter
    new_attempts = current_attempts + 1
    print(f"Preparing refinement attempt {new_attempts}/{max_attempts}")

    # Extract refinement suggestions from validation
    refinement_suggestions = []
    validation_result = state.get("validation_result")
    if validation_result:
        suggestions = validation_result.get("refinement_suggestions", [])
        refinement_suggestions = suggestions
        print(f"Applying {len(refinement_suggestions)} refinement "
              f"suggestions:")
        for i, suggestion in enumerate(refinement_suggestions[:3], 1):
            print(f"  {i}. {suggestion}")

    return {
        **state,
        "refinement_attempts": new_attempts,
        "workflow_step": "table_selection",
        "messages": state.get("messages", []) + [
            HumanMessage(
                content=(
                    f"Starting refinement attempt "
                    f"{new_attempts}/{max_attempts}"
                )
            )
        ]
    }


def completion_node(state: WorkflowState) -> WorkflowState:
    """
    COMPLETION: Finalize workflow with results and visualization
    """
    print("\n=== WORKFLOW COMPLETION ===")

    validation_status = state.get("validation_status", "unknown")
    refinement_attempts = state.get("refinement_attempts", 0)

    print(f"Final validation status: {validation_status}")
    print(f"Total refinement attempts: {refinement_attempts}")
    print(f"Steps completed: {', '.join(state.get('steps_completed', []))}")

    # Generate final mapping visualization
    if state.get("selected_table") and state.get("rag_results"):
        print("Generating comprehensive mapping visualization...")

        from tools import generate_mapping_visualization

        try:
            visualization = generate_mapping_visualization.invoke({
                "rag_results": json.dumps(state["rag_results"]),
                "table_selection": json.dumps(state["selected_table"]),
                "file_analysis": state.get("file_analysis_result")
            })

            visualization_data = json.loads(visualization)

            # Determine final workflow status
            if validation_status == "proceed_ingestion":
                workflow_status = "validation_passed"
                ready_for_execution = True
                print("Validation PASSED - Ready for database ingestion")
            elif validation_status == "manual_review":
                workflow_status = "requires_manual_review"
                ready_for_execution = False
                print("Manual review REQUIRED before ingestion")
            else:
                workflow_status = "completed"
                ready_for_execution = False
                print(f"Workflow completed with validation status: "
                      f"{validation_status}")

            return {
                **state,
                "workflow_status": workflow_status,
                "workflow_step": "completed",
                "mapping_visualization": visualization_data,
                "ready_for_execution": ready_for_execution,
                "ready_for_review": True,
                "messages": state.get("messages", []) + [
                    HumanMessage(
                        content=f"Workflow completed with status: "
                                f"{workflow_status}"
                )
                ]
            }

        except Exception as e:
            print(f"Warning: Failed to generate visualization: {str(e)}")

            return {
                **state,
                "workflow_status": "completed_without_visualization",
                "workflow_step": "completed",
                "ready_for_execution": False,
                "ready_for_review": True
            }

    # Fallback if no results available
    return {
        **state,
        "workflow_status": "completed_incomplete",
        "workflow_step": "completed",
        "ready_for_execution": False,
        "ready_for_review": False
    }


# ============================================================================

def should_continue_to_rag(
    state: WorkflowState
) -> Literal["continue", "error"]:
    """Decide whether to continue from file analysis to RAG matching"""
    if state.get("workflow_step") == "error":
        return "error"
    if not state.get("file_analysis_result"):
        return "error"
    file_analysis = state.get("file_analysis_result", "")
    if file_analysis and file_analysis.startswith("Error"):
        return "error"
    return "continue"


def should_continue_to_table_selection(
    state: WorkflowState
) -> Literal["continue", "error", "no_tables"]:
    """Decide whether to continue from RAG to table selection"""
    if state.get("workflow_step") == "error":
        return "error"

    rag_results = state.get("rag_results")
    if rag_results is not None:
        top_tables = rag_results.get("top_10_tables", [])
    else:
        top_tables = []

    if not top_tables:
        print("No matching tables found")
        return "no_tables"

    return "continue"


def should_continue_to_validation(
    state: WorkflowState
) -> Literal["continue", "error"]:
    """Decide whether to continue from table selection to validation"""
    if state.get("workflow_step") == "error":
        return "error"
    if not state.get("selected_table"):
        return "error"
    return "continue"


def should_refine_or_complete(
    state: WorkflowState
) -> Literal["complete", "refine", "manual_review", "error"]:
    """
    Decide whether to refine mapping, complete, or require manual review
    """
    if state.get("workflow_step") == "error":
        return "error"

    validation_status = state.get("validation_status")
    refinement_attempts = state.get("refinement_attempts", 0)
    max_refinements = state.get("max_refinements", 3)

    print("\n=== ROUTING DECISION ===")
    print(f"Validation status: {validation_status}")
    print(f"Refinement attempts: {refinement_attempts}/{max_refinements}")

    if validation_status == "proceed_ingestion":
        print("Decision: COMPLETE (validation passed)")
        return "complete"
    elif validation_status == "manual_review":
        print("Decision: MANUAL_REVIEW (critical issues detected)")
        return "manual_review"
    elif validation_status == "refine_mapping":
        if refinement_attempts < max_refinements:
            next_attempt = refinement_attempts + 1
            print(f"Decision: REFINE (attempt {next_attempt}/"
                  f"{max_refinements})")
            return "refine"
        else:
            print("Decision: MANUAL_REVIEW (max refinements reached)")
            return "manual_review"
    else:
        print(f"Decision: ERROR (unknown validation status: "
              f"{validation_status})")
        return "error"


# ============================================================================

def _classify_error(error_message: Optional[str]) -> str:
    """Classify error type based on error message"""
    if error_message is None:
        return "unknown_error"

    error_lower = error_message.lower()

    if "file not found" in error_lower or "no such file" in error_lower:
        return "file_not_found"
    elif "json" in error_lower and "parse" in error_lower:
        return "json_parsing_error"
    elif "rag" in error_lower or "vector" in error_lower:
        return "rag_system_error"
    elif "connection" in error_lower or "network" in error_lower:
        return "network_error"
    elif "permission" in error_lower or "access denied" in error_lower:
        return "permission_error"
    else:
        return "unknown_error"


def _generate_recovery_suggestions(
    error_message: str, failed_step: str
) -> list:
    """Generate recovery suggestions based on error type and failed step"""
    error_type = _classify_error(error_message)
    suggestions = []

    if error_type == "file_not_found":
        suggestions.extend([
            "Verify the file path is correct and the file exists",
            "Check file permissions and accessibility",
            "Ensure the file path uses the correct path separator for "
            "your OS"
        ])
    elif error_type == "json_parsing_error":
        suggestions.extend([
            "Check if the tool returned valid JSON format",
            "Verify the tool completed successfully before parsing",
            "Review tool output for unexpected formatting"
        ])
    elif error_type == "rag_system_error":
        suggestions.extend([
            "Verify RAG vector store is initialized and accessible",
            "Check QDRANT_URL and QDRANT_API_KEY environment variables",
            "Ensure the collection 'maxo_vector_store_v2' exists",
            "Run RAG system in 'feed' mode if collection is missing"
        ])
    elif error_type == "network_error":
        suggestions.extend([
            "Check your internet connection",
            "Verify API endpoints are accessible",
            "Check firewall settings"
        ])
    elif error_type == "permission_error":
        suggestions.extend([
            "Verify you have read/write permissions for the file",
            "Run with appropriate user permissions",
            "Check file is not locked by another process"
        ])
    else:
        suggestions.extend([
            f"Review the error message for {failed_step} step",
            "Check logs for additional context",
            "Verify all required environment variables are set",
            "Ensure all dependencies are installed"
        ])

    return suggestions


# ============================================================================

# def build_workflow_graph():
#     """Build the complete LangGraph workflow for database ingestion"""
#     print("Building LangGraph workflow for database ingestion...")

#     builder = StateGraph(WorkflowState)

#     # Add workflow nodes
#     builder.add_node("file_analysis", file_analysis_node)
#     builder.add_node("rag_matching", rag_matching_node)
#     builder.add_node("table_selection", table_selection_node)
#     builder.add_node("validation", validation_node)
#     builder.add_node("error_handler", error_handler_node)
#     builder.add_node("refinement", refinement_node)
#     builder.add_node("completion", completion_node)

#     # Define edges
#     builder.add_edge(START, "file_analysis")

#     # Conditional routing from file_analysis
#     builder.add_conditional_edges(
#         "file_analysis",
#         should_continue_to_rag,
#         {
#             "continue": "rag_matching",
#             "error": "error_handler"
#         }
#     )

#     # Conditional routing from rag_matching
#     builder.add_conditional_edges(
#         "rag_matching",
#         should_continue_to_table_selection,
#         {
#             "continue": "table_selection",
#             "error": "error_handler",
#             "no_tables": "error_handler"
#         }
#     )

#     # Conditional routing from table_selection
#     builder.add_conditional_edges(
#         "table_selection",
#         should_continue_to_validation,
#         {
#             "continue": "validation",
#             "error": "error_handler"
#         }
#     )

#     # Conditional routing from validation (main decision point)
#     builder.add_conditional_edges(
#         "validation",
#         should_refine_or_complete,
#         {
#             "complete": "completion",
#             "refine": "refinement",
#             "manual_review": "completion",
#             "error": "error_handler"
#         }
#     )

#     # Refinement loop back to table_selection
#     builder.add_edge("refinement", "table_selection")

#     # Terminal nodes
#     builder.add_edge("completion", END)
#     builder.add_edge("error_handler", END)

#     print("LangGraph workflow compiled successfully!")
#     return builder.compile()


# ============================================================================

if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(
        description="Database Ingestion Agent with LangGraph Workflow"
    )
    parser.add_argument(
        "file", nargs="?",
        help="File path to analyze and map to database"
    )
    parser.add_argument(
        "--table-name", dest="table_name",
        help="Preferred table name override", default=None
    )
    parser.add_argument(
        "--context", dest="user_context",
        help="Optional user context for ingestion", default=None
    )
    parser.add_argument(
        "--max-refinements", dest="max_refinements", type=int, default=3,
        help="Maximum refinement attempts (default: 3)"
    )
    args = parser.parse_args()

    if not args.file:
        print("Usage: python agent.py <file_path> [--table-name NAME] "
              "[--context CONTEXT] [--max-refinements N]")
        print("\nExample:")
        print("  python agent.py ../oppo_combi.csv")
        print("  python agent.py ../oppo_combi.csv --context "
              "'Sales opportunity data'")
        print("  python agent.py ../oppo_combi.csv --table-name Opportunity "
              "--max-refinements 5")
        sys.exit(1)

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print("=== DATABASE INGESTION WORKFLOW ===")
    print(f"File: {file_path}")
    print(f"Table preference: {args.table_name or 'None'}")
    print(f"User context: {args.user_context or 'Auto-generated'}")
    print(f"Max refinements: {args.max_refinements}")
    print("=" * 50)
    print()

    # Build the workflow graph
    graph = build_workflow_graph()

    # Prepare initial state
    initial_state = {
        "file_path": file_path,
        "user_context": args.user_context,
        "table_preference": args.table_name,
        "messages": [],
        "workflow_step": "start",
        "refinement_attempts": 0,
        "max_refinements": args.max_refinements,
        "errors": [],
        "steps_completed": [],
        "refinement_history": []
    }

    # Execute the workflow
    try:
        print("Starting workflow execution...")
        final_state = graph.invoke(initial_state)

        print("\n" + "=" * 50)
        print("=== WORKFLOW RESULTS ===")
        print("=" * 50)
        print(f"Status: {final_state.get('workflow_status', 'unknown')}")
        steps = final_state.get('steps_completed', [])
        print(f"Steps completed: {', '.join(steps)}")
        attempts = final_state.get('refinement_attempts', 0)
        print(f"Refinement attempts: {attempts}")
        ready_exec = final_state.get('ready_for_execution', False)
        print(f"Ready for execution: {ready_exec}")
        print(f"Requires review: "
              f"{final_state.get('ready_for_review', False)}")

        if final_state.get("errors"):
            print(f"\nErrors encountered: {len(final_state['errors'])}")
            for error in final_state["errors"]:
                print(f"  - {error}")

        # Print selected table info if available
        if final_state.get("selected_table"):
            selected = final_state["selected_table"].get("selected_table", {})
            print(f"\nSelected Table: {selected.get('table_name', 'Unknown')}")
            print(f"  Confidence: {selected.get('confidence_score', 0.0):.2f}")
            mappings = final_state['selected_table'].get('field_mappings', [])
            print(f"  Field mappings: {len(mappings)}")
            unmapped = final_state['selected_table'].get('unmapped_fields', [])
            print(f"  Unmapped fields: {len(unmapped)}")

        # Save results to JSON
        basename = os.path.basename(file_path)
        output_file = f"workflow_results_{basename}.json"
        with open(output_file, 'w') as f:
            # Convert state to JSON-serializable format
            json_state = {
                k: v for k, v in final_state.items()
                if k != "messages"
            }
            json.dump(json_state, f, indent=2, default=str)
        print(f"\nFull results saved to: {output_file}")

    except Exception as e:
        print(f"\nWorkflow execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
